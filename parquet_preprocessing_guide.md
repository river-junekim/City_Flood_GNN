# Parquet 형식 안내 및 도심 침수 예측 데이터 전처리 가이드

**프로젝트**: 도심 침수 예측 GNN — 서울시 하수관로·도로노면 수위 센서 데이터  
**작성 목적**: 원본 CSV/TXT → Parquet 전환 배경과 전처리 파이프라인 공유

---

## 1. Parquet 형식이란?

Apache Parquet은 **열(Column) 기반 바이너리 저장 형식**입니다. CSV/TXT가 행 단위로 데이터를 순서대로 저장하는 것과 달리, Parquet은 같은 컬럼의 값들을 연속적으로 모아서 저장합니다. 원래 Hadoop 생태계(Spark, Hive 등)를 위해 설계되었으나, 현재는 `pandas` / `PyArrow`를 통해 Python 환경에서도 표준처럼 사용됩니다.

```
[CSV - 행 기반]                  [Parquet - 열 기반]
sensor_id, timestamp, level      sensor_id 컬럼: [A, A, B, B, ...]
A, 2024-01-01 00:00, 1.2    →    timestamp 컬럼: [00:00, 00:01, 00:00, ...]
A, 2024-01-01 00:01, 1.3         level     컬럼: [1.2, 1.3, 0.5, 0.6, ...]
B, 2024-01-01 00:00, 0.5
B, 2024-01-01 00:01, 0.6
```

---

## 2. CSV/TXT 대비 장단점

### 장점

| 항목 | CSV/TXT | Parquet |
|------|---------|---------|
| **파일 크기** | 텍스트 그대로 저장 | 컬럼별 delta/dictionary 인코딩 + Snappy 압축으로 보통 **3~10배 작음** |
| **읽기 속도** | 전체 파일 파싱 필요 | 필요한 컬럼만 선택 로드 가능 (Column Pruning) |
| **타입 정보** | 없음 (모든 값이 문자열) | 스키마가 파일에 내장 — `float64`, `timestamp[ns]`, `int8` 등 정확히 보존 |
| **필터 푸시다운** | 불가 | `filters=[('timestamp','>=', ...)]`로 읽기 전에 행 스킵 가능 |
| **대용량 처리** | pandas가 전체를 메모리에 올려야 함 | `ParquetWriter`로 스트리밍/청크 단위 쓰기 가능 |
| **한글 인코딩** | cp949/utf-8 혼재 → 매번 감지 필요 | UTF-8 내장, 인코딩 문제 없음 |

### 단점

| 항목 | 설명 |
|------|------|
| **사람이 직접 열람 불가** | 바이너리 형식이라 텍스트 에디터로 볼 수 없음. `pd.read_parquet()` 또는 DuckDB 등 필요 |
| **범용성** | Excel, 메모장으로 열 수 없어 비개발자와의 공유에 불편 |
| **행 단위 추가(Append)** | CSV는 파일 끝에 행 추가가 쉽지만, Parquet은 Row Group 구조상 재작성이 필요 |
| **라이브러리 의존** | `pyarrow` 또는 `fastparquet` 패키지 설치 필요 |

### 빠른 읽기 예시

```python
import pandas as pd

# 특정 기간만 로드 (전체 파일을 메모리에 올리지 않음)
df = pd.read_parquet(
    'sewer_cleaned.parquet',
    filters=[('timestamp', '>=', '2024-06-01'),
             ('timestamp', '<',  '2024-10-01')],
    columns=['sensor_id', 'timestamp', 'level']
)
```

---

## 3. 전처리 파이프라인 개요

```
원본 파일                           최종 출력
sewer_csv/ (CSV, 44개)  ─┐
road_txt/  (TXT, 24개)  ─┘
                          Step 01. CSV/TXT → Parquet
                          Step 02. 스키마 통일 및 통합
                          Step 03. 이상치 제거 + 결측치 보간
                          Step 04. 제원표 조인 (메타데이터)
                          Step 05. 상관 분석 (센서 쌍 구성)
                          Step 06. 파생 Feature 생성
                          Step 07. 인접행렬 (Gaussian 가중치)
                          Step 08. 공통 기간 분리
                          Step 09. 정규화 (leakage 방지)
                          Step 10. Train / Val / Test 분리
                          Step 11. GNN 설정 파일 생성
                                ↓
                     gnn_config.json
                     train/ val/ test/  ← 학습 준비 완료
```

**데이터 규모**

| 구분 | 기간 | 센서 수 | 파일 형식 |
|------|------|---------|-----------|
| 하수관로 수위 | 2022-01 ~ 2025-08 (44개월) | 456개 노드 | CSV → Parquet |
| 도로노면 수위 | 2024-01 ~ 2025-12 (24개월) | 112개 노드 | TXT → Parquet |
| 공통 학습 기간 | 2024-01 ~ 2025-08 (20개월) | — | — |

---

## 4. Step별 코드 상세

### Step 01. 원본 CSV/TXT → Parquet 변환

**문제**: 연도마다 인코딩(`cp949` / `utf-8-sig` / `utf-8`)과 컬럼명(한글/영문)이 달라서 단순 읽기 불가  
**해결**: 인코딩 자동 감지 + `COLUMN_MAP`으로 한영 컬럼명 통일

```python
import pandas as pd
import glob, os

COLUMN_MAP = {
    '고유번호'  : 'sensor_id',  '?고유번호' : 'sensor_id',
    '측정일자'  : 'timestamp',  '측정수위'  : 'level',
    '통신상태'  : 'comm_status','unq_no'    : 'sensor_id',
    'msrmt_ymd' : 'timestamp',  'msrmt_watl': 'level',
    'sgn_stts'  : 'comm_status',
}

def read_sewer_file(f):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns=COLUMN_MAP)
            if 'sensor_id' in df.columns and 'timestamp' in df.columns:
                return df, enc
        except Exception:
            continue
    return None, None

def save_parquet(df, out_path):
    df['timestamp']   = pd.to_datetime(df['timestamp']).astype('datetime64[ns]')
    df['sensor_id']   = df['sensor_id'].astype(str)
    df['level']       = pd.to_numeric(df['level'], errors='coerce')
    if 'comm_status' in df.columns:
        df['comm_status'] = df['comm_status'].astype(str)
    df.reset_index(drop=True).to_parquet(out_path, index=False, engine='pyarrow')

def convert_sewer(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    os.makedirs('./dataset/processed/raw_parquet/sewer', exist_ok=True)
    for f in files:
        fname    = os.path.splitext(os.path.basename(f))[0]
        out_path = f'./dataset/processed/raw_parquet/sewer/{fname}.parquet'
        if os.path.exists(out_path):
            print(f'SKIP: {fname}'); continue
        df, enc = read_sewer_file(f)
        if df is None:
            print(f'ERROR: {fname}'); continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        save_parquet(df, out_path)
        print(f'OK [{enc}]: {fname} → {len(df):,}행')

def convert_road(txt_dir):
    files = sorted(glob.glob(os.path.join(txt_dir, '*.txt')))
    os.makedirs('./dataset/processed/raw_parquet/road', exist_ok=True)
    skip = ['2023년 1~12월.txt']
    for f in files:
        fname    = os.path.splitext(os.path.basename(f))[0]
        out_path = f'./dataset/processed/raw_parquet/road/{fname}.parquet'
        if os.path.basename(f) in skip:
            print(f'SKIP(부족): {fname}'); continue
        if os.path.exists(out_path):
            print(f'SKIP: {fname}'); continue
        try:
            df = pd.read_csv(f, sep='\t', encoding='cp949')
            df = df.rename(columns={'ROADGAUGE_NAME':'sensor_id',
                                    'DATA_TIME':'timestamp','LEVEL_DATA':'level'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            save_parquet(df, out_path)
            print(f'OK: {fname} → {len(df):,}행')
        except Exception as e:
            print(f'ERROR: {fname} → {e}')

convert_sewer('./sewer_csv/')
convert_road('./road_txt/')
```

---

### Step 02. 스키마 통일 및 통합

**문제**: 2025-07/08 신규 파일에 기존에 없던 컬럼(`se_cd`, `pstn_info`)이 추가됨  
**해결**: `ParquetWriter`에 고정 스키마를 지정하여 어떤 파일이든 기준 컬럼만 유지

```python
import glob, os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEWER_RENAME = {
    'se_cd'    : '구분코드',
    'se_nm'    : '구분명',
    'pstn_info': '위치정보',  # 기준 스키마에 없으므로 제거됨
}

SEWER_BASE_COLS = ['sensor_id', '구분코드', '구분명', 'timestamp', 'level', 'comm_status']
ROAD_BASE_COLS  = ['sensor_id', 'timestamp', 'level']

SEWER_SCHEMA = pa.schema([
    ('sensor_id',  pa.string()),  ('구분코드', pa.string()),
    ('구분명',     pa.string()),  ('timestamp', pa.timestamp('ns')),
    ('level',      pa.float64()), ('comm_status', pa.string()),
])
ROAD_SCHEMA = pa.schema([
    ('sensor_id', pa.string()),  ('timestamp', pa.timestamp('ns')),
    ('level',     pa.float64()),
])

def align_schema(df, rename_map, base_cols):
    df = df.rename(columns=rename_map)
    for c in base_cols:
        if c not in df.columns: df[c] = None
    df = df[base_cols].copy()
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').astype('datetime64[ns]')
    df['level']     = pd.to_numeric(df['level'], errors='coerce')
    if 'comm_status' in df.columns:
        df['comm_status'] = df['comm_status'].astype(str)
    if '구분코드' in df.columns:
        df['구분코드'] = df['구분코드'].astype(str)
    if '구분명' in df.columns:
        df['구분명'] = df['구분명'].astype(str)
    return df

def merge_parquets(raw_dir, out_path, rename_map, base_cols, schema):
    files  = sorted(glob.glob(os.path.join(raw_dir, '*.parquet')))
    writer = pq.ParquetWriter(out_path, schema)
    total  = 0
    for f in files:
        df    = pd.read_parquet(f)
        df    = align_schema(df, rename_map, base_cols)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)
        total += len(df)
        print(f'  {os.path.basename(f)} → {len(df):,}행 (누적 {total:,}행)')
        del df, table
    writer.close()
    print(f'완료: {out_path} / 총 {total:,}행')

os.makedirs('./dataset/processed/merged', exist_ok=True)
merge_parquets('./dataset/processed/raw_parquet/sewer/',
               './dataset/processed/merged/sewer_all.parquet', SEWER_RENAME, SEWER_BASE_COLS, SEWER_SCHEMA)
merge_parquets('./dataset/processed/raw_parquet/road/',
               './dataset/processed/merged/road_all.parquet',  {}, ROAD_BASE_COLS, ROAD_SCHEMA)
```

---

### Step 03. 이상치 제거 + 결측치 보간 (병렬 처리)

**이상치 처리 기준**
- 하수관로: 음수 값, 제원표의 `max_level_m` 초과 값 → NaN
- 도로노면: 오류코드 `{312, 419, 999, 1000}` → NaN

**결측치 보간**: 갭 ≤ 10분이면 선형 보간, 그 이상은 NaN 유지 (과도한 보간 방지)  
**병렬화**: 하수관로 workers=2 (파일당 ~970MB), 도로노면 workers=4 (파일당 ~141MB)

```python
import os, glob, gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp

SEWER_RAW        = './dataset/processed/raw_parquet/sewer/'
ROAD_RAW         = './dataset/processed/raw_parquet/road/'
SEWER_CLEAN_DIR  = './dataset/processed/cleaned/sewer/'
ROAD_CLEAN_DIR   = './dataset/processed/cleaned/road/'
MERGED_DIR       = './dataset/processed/cleaned/'
GAP_LIMIT_MIN    = 10
ROAD_ERROR_CODES = {312, 419, 999, 1000}
WORKERS_SEWER    = 2
WORKERS_ROAD     = 4

SEWER_SCHEMA = pa.schema([('sensor_id',pa.string()),('timestamp',pa.timestamp('ns')),('level',pa.float64())])
ROAD_SCHEMA  = pa.schema([('sensor_id',pa.string()),('timestamp',pa.timestamp('ns')),('level',pa.float64())])

for d in [SEWER_CLEAN_DIR, ROAD_CLEAN_DIR, MERGED_DIR]:
    os.makedirs(d, exist_ok=True)

# 제원표에서 sensor별 측정범위 최댓값 로드
master = pd.read_excel('./dataset/processed/서울시 수위계(하수관로) 제원표_20260310.xlsx',
                       header=1, engine='openpyxl')
master.columns = [c.replace('\n',' ').strip() for c in master.columns]
max_map = (master[['수위계번호 (지점코드)','측정범위 최댓값(m)']]
           .rename(columns={'수위계번호 (지점코드)':'sid','측정범위 최댓값(m)':'mx'})
           .assign(sid=lambda d:d['sid'].astype(str),
                   mx=lambda d:pd.to_numeric(d['mx'],errors='coerce'))
           .dropna(subset=['mx']).set_index('sid')['mx'].to_dict())

# ── 보간 함수 ──────────────────────────────────────────────────────────────────
def _interpolate(df, gap_limit=GAP_LIMIT_MIN):
    parts = []
    for sid, grp in df.groupby('sensor_id', sort=False):
        grp = (grp[['timestamp','level']].sort_values('timestamp')
               .drop_duplicates('timestamp').set_index('timestamp'))
        grid = pd.date_range(grp.index.min(), grp.index.max(), freq='1min')
        grp  = grp.reindex(grid); grp.index.name = 'timestamp'
        grp['sensor_id'] = sid
        grp['level'] = grp['level'].interpolate(
            method='linear', limit=gap_limit,
            limit_direction='forward', limit_area='inside')
        parts.append(grp.reset_index()[['sensor_id','timestamp','level']])
    return pd.concat(parts, ignore_index=True)

# ── 파일 처리 함수 (fork 안전하도록 최상위 정의) ─────────────────────────────────
def process_sewer_file(args):
    f, _max_map, out_dir = args
    fname = os.path.splitext(os.path.basename(f))[0]
    out   = os.path.join(out_dir, f'{fname}.parquet')
    if os.path.exists(out): return f'SKIP: {fname}'
    try:
        df = pd.read_parquet(f, engine='pyarrow')
        neg  = int((df['level']<0).sum())
        df.loc[df['level']<0,'level'] = np.nan
        df['_max'] = df['sensor_id'].map(_max_map).fillna(20.0)
        over = int((df['level']>df['_max']).sum())
        df.loc[df['level']>df['_max'],'level'] = np.nan
        df.drop(columns=['_max'], inplace=True)
        df = _interpolate(df)
        df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
        pq.write_table(pa.Table.from_pandas(df, schema=SEWER_SCHEMA, preserve_index=False), out)
        del df; gc.collect()
        return f'OK: {fname} [음수:{neg}, max초과:{over}]'
    except Exception as e: return f'ERROR: {fname} → {e}'

def process_road_file(args):
    f, out_dir = args
    fname = os.path.splitext(os.path.basename(f))[0]
    out   = os.path.join(out_dir, f'{fname}.parquet')
    if os.path.exists(out): return f'SKIP: {fname}'
    try:
        df = pd.read_parquet(f, engine='pyarrow')
        err = int(df['level'].isin(ROAD_ERROR_CODES).sum())
        df.loc[df['level'].isin(ROAD_ERROR_CODES),'level'] = np.nan
        df = _interpolate(df)
        df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
        pq.write_table(pa.Table.from_pandas(df, schema=ROAD_SCHEMA, preserve_index=False), out)
        del df; gc.collect()
        return f'OK: {fname} [오류코드:{err}]'
    except Exception as e: return f'ERROR: {fname} → {e}'

def merge_cleaned(clean_dir, out_path, schema):
    files  = sorted(glob.glob(os.path.join(clean_dir,'*.parquet')))
    writer = pq.ParquetWriter(out_path, schema)
    total  = 0
    for f in files:
        table = pq.read_table(f, schema=schema)
        writer.write_table(table); total += table.num_rows; del table
    writer.close(); return total

# 하수관로 병렬 처리 (workers=2, 파일당 ~970MB)
ctx = mp.get_context('fork')
sewer_args = [(f, max_map, SEWER_CLEAN_DIR)
              for f in sorted(glob.glob(SEWER_RAW+'*.parquet'))]
print(f'하수관로 {len(sewer_args)}개 파일 처리 (workers={WORKERS_SEWER})')
with ctx.Pool(processes=WORKERS_SEWER, maxtasksperchild=1) as pool:
    for msg in pool.imap_unordered(process_sewer_file, sewer_args):
        print(f'  {msg}')

# 도로노면 병렬 처리 (workers=4, 파일당 ~141MB)
road_args = [(f, ROAD_CLEAN_DIR)
             for f in sorted(glob.glob(ROAD_RAW+'*.parquet'))]
print(f'\n도로노면 {len(road_args)}개 파일 처리 (workers={WORKERS_ROAD})')
with ctx.Pool(processes=WORKERS_ROAD, maxtasksperchild=1) as pool:
    for msg in pool.imap_unordered(process_road_file, road_args):
        print(f'  {msg}')

# 월별 파일 → 통합 parquet
n = merge_cleaned(SEWER_CLEAN_DIR, MERGED_DIR+'sewer_cleaned.parquet', SEWER_SCHEMA)
print(f'\nsewer_cleaned.parquet: {n:,}행')
n = merge_cleaned(ROAD_CLEAN_DIR,  MERGED_DIR+'road_cleaned.parquet',  ROAD_SCHEMA)
print(f'road_cleaned.parquet: {n:,}행')
```

---

### Step 04. 제원표 조인 (노드 메타데이터)

**문제**: 도로노면 파일의 `sensor_id`는 "관악구 신림동 교차로" 형태의 지점명인데, 제원표에는 "신림동 교차로"로 저장되어 있어 단순 조인 불가  
**해결**: 정규화 + 자치구 접두사 제거로 2단계 매핑, 그래도 매칭 안 되면 unmatched 표기

```python
import re
import numpy as np
import pandas as pd
import os

OUT_DIR = './dataset/processed/cleaned/'

# ── 제원표 로드 ────────────────────────────────────────────────────────────────
road_master = pd.read_excel('./dataset/processed/서울시 수위계(도로) 제원표_20260310.xlsx',
                            header=1, engine='openpyxl')
road_master.columns = [c.replace('\n',' ').strip() for c in road_master.columns]
road_master = road_master.rename(columns={
    '수위계번호 (지점코드)':'sensor_code', '지점명':'지점명', '배수구역':'배수구역',
    '자치구':'자치구', '위경도 lon.':'lat', '위경도 lat.':'lon',
    '측정범위 최댓값(cm)':'max_level_cm', '기왕 최대값(cm)':'hist_max_cm',
    '수위계 상태 (정상/미수신/정비중 등)':'status', '자료관측주기 (1분/10분)':'obs_interval'})
road_master['max_level_cm'] = pd.to_numeric(road_master['max_level_cm'], errors='coerce')

sewer_master = pd.read_excel('./dataset/processed/서울시 수위계(하수관로) 제원표_20260310.xlsx',
                             header=1, engine='openpyxl')
sewer_master.columns = [c.replace('\n',' ').strip() for c in sewer_master.columns]
sewer_master = sewer_master.rename(columns={
    '수위계번호 (지점코드)':'sensor_id', '지점명':'지점명', '배수구역':'배수구역',
    '자치구':'자치구', '관규격':'관규격', '위경도 lon.':'lat', '위경도 lat.':'lon',
    '측정범위 최댓값(m)':'max_level_m', '기왕 최대값(m)':'hist_max_m',
    '수위계 상태 (정상/미수신/정비중 등)':'status', '자료관측주기 (1분/10분)':'obs_interval'})
sewer_master['sensor_id']   = sewer_master['sensor_id'].astype(str)
sewer_master['max_level_m'] = pd.to_numeric(sewer_master['max_level_m'], errors='coerce')
sewer_master['lat'] = pd.to_numeric(sewer_master['lat'], errors='coerce')
sewer_master['lon'] = pd.to_numeric(sewer_master['lon'], errors='coerce')

# ── 관규격 파싱 → pipe_height_m ────────────────────────────────────────────────
def parse_pipe_height_m(spec):
    if pd.isna(spec): return np.nan
    s = str(spec).strip()
    m = re.match(r'[Øø](\d+)', s)
    if m: return float(m.group(1))/1000
    m = re.search(r'(\d+\.?\d*)X(\d+\.?\d*)', s, re.I)
    if m: return min(float(m.group(1)), float(m.group(2)))
    m = re.match(r'(\d+\.?\d*)$', s)
    if m: return float(m.group(1))
    return np.nan

sewer_master['pipe_height_m'] = sewer_master['관규격'].apply(parse_pipe_height_m)
print(f'관규격 파싱: {sewer_master["pipe_height_m"].notna().sum()}/{len(sewer_master)}개')

# ── sensor_id 매핑 (지점명 불일치 해결) ───────────────────────────────────────
def norm(s):         return re.sub(r'\s+','',str(s)).strip()
def strip_prefix(s): return re.sub(r'^[가-힣]+[시구군]\s+','',str(s)).strip()

m_exact = {norm(r['지점명']): r for _,r in road_master.iterrows()}
m_strip = {norm(strip_prefix(r['지점명'])): r for _,r in road_master.iterrows()}

road_pq_ids = (pd.read_parquet(OUT_DIR+'road_cleaned.parquet',
                               engine='pyarrow', columns=['sensor_id'])
               ['sensor_id'].unique())

mapping_rows, unmatched = [], []
for uid in road_pq_ids:
    n = norm(uid)
    if n in m_exact:
        row = m_exact[n].copy(); row['sensor_id']=uid; row['match_method']='exact'
        mapping_rows.append(row)
    elif n in m_strip:
        row = m_strip[n].copy(); row['sensor_id']=uid; row['match_method']='prefix_strip'
        mapping_rows.append(row)
    else:
        unmatched.append(uid)

road_map     = pd.DataFrame(mapping_rows)
road_missing = pd.DataFrame({'sensor_id':unmatched,'match_method':'unmatched'})
road_node    = pd.concat([road_map, road_missing], ignore_index=True)

print(f'정확 매핑:   {(road_node["match_method"]=="exact").sum()}개')
print(f'접두사 제거: {(road_node["match_method"]=="prefix_strip").sum()}개')
print(f'미매핑:      {(road_node["match_method"]=="unmatched").sum()}개')

NODE_COLS_SEWER = ['sensor_id','지점명','배수구역','자치구','lat','lon',
                   '관규격','pipe_height_m','max_level_m','hist_max_m','status','obs_interval']
NODE_COLS_ROAD  = ['sensor_id','sensor_code','지점명','배수구역','자치구','lat','lon',
                   'max_level_cm','hist_max_cm','status','obs_interval','match_method']

sewer_actual = set(pd.read_parquet(OUT_DIR+'sewer_cleaned.parquet',
                                   engine='pyarrow',columns=['sensor_id'])['sensor_id'].unique())
sewer_node = sewer_master[NODE_COLS_SEWER].query('sensor_id in @sewer_actual').reset_index(drop=True)
road_node  = road_node[[c for c in NODE_COLS_ROAD if c in road_node.columns]].copy()

sewer_node.to_parquet(OUT_DIR+'sewer_node.parquet', index=False, engine='pyarrow')
road_node.to_parquet(OUT_DIR+'road_node.parquet',   index=False, engine='pyarrow')
print(f'sewer_node: {len(sewer_node)}개 / road_node: {len(road_node)}개')
```

---

### Step 05. 탐색적 분석 + 상관 분석

**Parquet 필터 푸시다운 활용**: 전체 파일을 메모리에 올리지 않고 여름철(6~9월) 장마 구간만 선택적으로 로드  
**센서 쌍 구성**: 배수구역 일치 + Haversine 거리 ≤ 1km  
**교차 상관**: ±60분(6스텝) 범위에서 최적 lag와 Pearson 상관계수를 전체/이벤트(수위>0) 구분하여 계산

```python
import numpy as np
import pandas as pd
import glob, os
from math import radians, sin, cos, sqrt, atan2

sewer_node = pd.read_parquet('./dataset/processed/cleaned/sewer_node.parquet', engine='pyarrow')
road_node  = pd.read_parquet('./dataset/processed/cleaned/road_node.parquet',  engine='pyarrow')

# 여름철(장마) 데이터만 로드 — Parquet filter pushdown 활용
PERIODS = [(pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-01')),
           (pd.Timestamp('2025-06-01'), pd.Timestamp('2025-10-01'))]

def load_period(path, s, e):
    return pd.read_parquet(path, engine='pyarrow',
                           filters=[('timestamp','>=',s),('timestamp','<',e)],
                           columns=['sensor_id','timestamp','level'])

sewer_raw = pd.concat([load_period('./dataset/processed/cleaned/sewer_cleaned.parquet',s,e) for s,e in PERIODS])
road_raw  = pd.concat([load_period('./dataset/processed/cleaned/road_cleaned.parquet', s,e) for s,e in PERIODS])

# 10분 집계 (하수: 평균 / 도로: 최대)
sewer_10 = (sewer_raw.set_index('timestamp').groupby('sensor_id')['level']
            .resample('10min').mean().reset_index().rename(columns={'level':'sewer_level'}))
road_10  = (road_raw.set_index('timestamp').groupby('sensor_id')['level']
            .resample('10min').max().reset_index().rename(columns={'level':'road_level'}))
print(f'sewer 10분: {len(sewer_10):,}행 / road 10분: {len(road_10):,}행')

# ── 센서 쌍 구성 (배수구역 일치 + 1km 이내) ───────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000; d = lambda x: radians(x)
    dlat, dlon = d(lat2-lat1), d(lon2-lon1)
    a = sin(dlat/2)**2 + cos(d(lat1))*cos(d(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

DIST_THRESHOLD_M = 1000
sewer_geo = sewer_node.dropna(subset=['lat','lon'])
road_geo  = road_node.dropna(subset=['lat','lon'])

pairs = []
for district, s_grp in sewer_geo.groupby('배수구역'):
    r_grp = road_geo[road_geo['배수구역']==district]
    if r_grp.empty: continue
    for _,sr in s_grp.iterrows():
        for _,rr in r_grp.iterrows():
            d = haversine(sr['lat'],sr['lon'],rr['lat'],rr['lon'])
            if d <= DIST_THRESHOLD_M:
                pairs.append({'sewer_id':sr['sensor_id'],'sewer_name':sr['지점명'],
                              'road_id':rr['sensor_id'],'road_name':rr['지점명'],
                              '배수구역':district,'자치구':sr['자치구'],
                              'distance_m':round(d,1),
                              'sewer_lat':sr['lat'],'sewer_lon':sr['lon'],
                              'road_lat':rr['lat'],'road_lon':rr['lon']})

pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=['sewer_id','road_id'])
print(f'유효 센서 쌍: {len(pairs_df)}개')

# ── 교차 상관 계산 ─────────────────────────────────────────────────────────────
MAX_LAG = 6; MIN_SAMPLES = 30

def cross_corr_best(x, y, max_lag=MAX_LAG):
    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag+1):
        xs   = x.shift(lag)
        mask = xs.notna() & y.notna()
        if mask.sum() < MIN_SAMPLES: continue
        c = xs[mask].corr(y[mask])
        if abs(c) > abs(best_corr): best_corr, best_lag = c, lag
    return best_lag, best_corr, (x.notna()&y.notna()).sum()

results = []
for _,row in pairs_df.iterrows():
    s_ts = sewer_10[sewer_10['sensor_id']==row['sewer_id']].set_index('timestamp')['sewer_level']
    r_ts = road_10[road_10['sensor_id']==row['road_id']].set_index('timestamp')['road_level']
    idx  = s_ts.index.union(r_ts.index)
    s_ts, r_ts = s_ts.reindex(idx), r_ts.reindex(idx)
    lag_all, corr_all, n_all = cross_corr_best(s_ts, r_ts)
    ev_mask = r_ts > 0
    if ev_mask.sum() >= MIN_SAMPLES:
        lag_ev, corr_ev, n_ev = cross_corr_best(s_ts[ev_mask], r_ts[ev_mask])
    else:
        lag_ev, corr_ev, n_ev = np.nan, np.nan, 0
    results.append({**row,
                    'corr_all':round(corr_all,4),'lag_all':lag_all,'n_all':n_all,
                    'corr_event':round(corr_ev,4) if not np.isnan(corr_ev) else np.nan,
                    'lag_event':lag_ev,'n_event':n_ev})

corr_df = pd.DataFrame(results)
corr_df.to_parquet('./dataset/processed/cleaned/correlation_results.parquet',
                   index=False, engine='pyarrow')
print(f'완료: {len(corr_df)}쌍 저장')
print(corr_df[['corr_all','corr_event']].describe().round(3))
```

---

### Step 06. 파생 Feature 생성

하수관로 9개, 도로노면 10개 feature를 월별 파일 단위로 처리하여 메모리 사용 최소화

| Feature | 하수관로 | 도로노면 |
|---------|---------|---------|
| `level_diff` | 이전 타임스텝 대비 수위 변화량 | 동일 |
| `fill_rate` | 수위 / 관 높이 (만수율, 0~1) | — |
| `flood_flag` | — | 수위 > 0 이면 1 |
| `flood_stage` | — | 0=정상, 1~4단계 |
| `hour_sin/cos` | 시간 cyclic 인코딩 | 동일 |
| `month_sin/cos` | 월 cyclic 인코딩 | 동일 |
| `season` | 계절 (1=봄, 2=여름, 3=가을, 4=겨울) | 동일 |
| `is_weekend` | 주말 여부 | 동일 |

```python
import pandas as pd, numpy as np, glob, os, gc
import pyarrow as pa, pyarrow.parquet as pq

sewer_node = pd.read_parquet('./dataset/processed/cleaned/sewer_node.parquet', engine='pyarrow')
road_node  = pd.read_parquet('./dataset/processed/cleaned/road_node.parquet',  engine='pyarrow')

pipe_map  = sewer_node.dropna(subset=['pipe_height_m']).set_index('sensor_id')['pipe_height_m'].to_dict()
maxlv_map = sewer_node.dropna(subset=['max_level_m']).set_index('sensor_id')['max_level_m'].to_dict()
SEASON_MAP = {12:4,1:4,2:4, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}

SEWER_FEAT_SCHEMA = pa.schema([
    ('sensor_id',pa.string()), ('timestamp',pa.timestamp('ns')),
    ('level',pa.float64()),    ('level_diff',pa.float64()),
    ('fill_rate',pa.float64()),('hour',pa.int8()),
    ('month',pa.int8()),       ('season',pa.int8()),('is_weekend',pa.int8())])
ROAD_FEAT_SCHEMA = pa.schema([
    ('sensor_id',pa.string()), ('timestamp',pa.timestamp('ns')),
    ('level',pa.float64()),    ('level_diff',pa.float64()),
    ('flood_flag',pa.int8()),  ('flood_stage',pa.int8()),
    ('hour',pa.int8()),        ('month',pa.int8()),
    ('season',pa.int8()),      ('is_weekend',pa.int8())])

os.makedirs('./dataset/processed/features/sewer/', exist_ok=True)
os.makedirs('./dataset/processed/features/road/',  exist_ok=True)

# ── 하수관로 ───────────────────────────────────────────────────────────────────
for f in sorted(glob.glob('./dataset/processed/cleaned/sewer/*.parquet')):
    fname = os.path.basename(f)
    out   = f'./dataset/processed/features/sewer/{fname}'
    if os.path.exists(out): print(f'SKIP: {fname}'); continue
    df = pd.read_parquet(f, engine='pyarrow').sort_values(['sensor_id','timestamp'])
    df['level_diff'] = df.groupby('sensor_id')['level'].diff().fillna(0)
    df['_d']         = df['sensor_id'].map(pipe_map).fillna(df['sensor_id'].map(maxlv_map))
    df['fill_rate']  = (df['level']/df['_d']).clip(0,1.0).astype('float64')
    df.drop(columns=['_d'], inplace=True)
    df['hour']       = df['timestamp'].dt.hour.astype('int8')
    df['month']      = df['timestamp'].dt.month.astype('int8')
    df['season']     = df['month'].map(SEASON_MAP).astype('int8')
    df['is_weekend'] = (df['timestamp'].dt.dayofweek>=5).astype('int8')
    df['level_diff'] = df['level_diff'].astype('float64')
    pq.write_table(pa.Table.from_pandas(df[SEWER_FEAT_SCHEMA.names],
                   schema=SEWER_FEAT_SCHEMA, preserve_index=False), out)
    print(f'OK: {fname} → {len(df):,}행'); del df; gc.collect()

# ── 도로노면 ───────────────────────────────────────────────────────────────────
for f in sorted(glob.glob('./dataset/processed/cleaned/road/*.parquet')):
    fname = os.path.basename(f)
    out   = f'./dataset/processed/features/road/{fname}'
    if os.path.exists(out): print(f'SKIP: {fname}'); continue
    df = pd.read_parquet(f, engine='pyarrow').sort_values(['sensor_id','timestamp'])
    df['level_diff']  = df.groupby('sensor_id')['level'].diff().fillna(0)
    df['flood_flag']  = (df['level']>0).astype('int8')
    df['flood_stage'] = pd.cut(df['level'].fillna(0), bins=[-1,0,5,20,50,9999],
                               labels=[0,1,2,3,4]).astype('int8')
    df['hour']        = df['timestamp'].dt.hour.astype('int8')
    df['month']       = df['timestamp'].dt.month.astype('int8')
    df['season']      = df['month'].map(SEASON_MAP).astype('int8')
    df['is_weekend']  = (df['timestamp'].dt.dayofweek>=5).astype('int8')
    df['level_diff']  = df['level_diff'].astype('float64')
    pq.write_table(pa.Table.from_pandas(df[ROAD_FEAT_SCHEMA.names],
                   schema=ROAD_FEAT_SCHEMA, preserve_index=False), out)
    print(f'OK: {fname} → {len(df):,}행'); del df; gc.collect()
```

---

### Step 07. 인접행렬 생성 (Gaussian 가중치)

**가중치 공식**: `w = exp(-d / σ)`, σ=300m, threshold=0.1  
**고립 도로 센서 처리**: 배수구역 조건을 만족하는 하수관로가 없는 경우, 2km 이내 최근접 하수관과 fallback 연결

```python
import pandas as pd, numpy as np, os
from math import radians, sin, cos, sqrt, atan2
import pyarrow.parquet as pq

corr_df    = pd.read_parquet('./dataset/processed/cleaned/correlation_results.parquet', engine='pyarrow')
sewer_node = pd.read_parquet('./dataset/processed/cleaned/sewer_node.parquet', engine='pyarrow')
road_node  = pd.read_parquet('./dataset/processed/cleaned/road_node.parquet',  engine='pyarrow')
SIGMA, THRESHOLD = 300, 0.1

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000; d = lambda x: radians(x)
    dlat, dlon = d(lat2-lat1), d(lon2-lon1)
    a = sin(dlat/2)**2 + cos(d(lat1))*cos(d(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# ── 기본 sewer→road 인접행렬 ─────────────────────────────────────────────────
adj = corr_df[['sewer_id','sewer_name','road_id','road_name','배수구역','자치구','distance_m']].copy()
adj['gauss_weight'] = np.exp(-adj['distance_m']/SIGMA)
adj = adj[adj['gauss_weight']>=THRESHOLD].reset_index(drop=True)
s_geo = sewer_node[['sensor_id','lat','lon']].rename(columns={'sensor_id':'sewer_id','lat':'s_lat','lon':'s_lon'})
r_geo = road_node[['sensor_id','lat','lon']].rename(columns={'sensor_id':'road_id','lat':'r_lat','lon':'r_lon'})
adj   = adj.merge(s_geo,on='sewer_id',how='left').merge(r_geo,on='road_id',how='left')
adj['edge_type'] = 'original'

# ── fallback 엣지 (고립 도로 센서 → 2km 이내 최근접 하수관) ─────────────────
isolated  = road_node[~road_node['sensor_id'].isin(adj['road_id'])].dropna(subset=['lat','lon'])
sewer_geo = sewer_node.dropna(subset=['lat','lon'])
new_rows  = []
for _,r in isolated.iterrows():
    best_d, best_s = 9999999, None
    for _,s in sewer_geo.iterrows():
        d = haversine(r['lat'],r['lon'],s['lat'],s['lon'])
        if d < best_d: best_d, best_s = d, s
    if best_d <= 2000 and best_s is not None:
        new_rows.append({'sewer_id':best_s['sensor_id'],'sewer_name':best_s['지점명'],
                         'road_id':r['sensor_id'],'road_name':r['지점명'],
                         '배수구역':best_s.get('배수구역',''),'자치구':best_s.get('자치구',''),
                         'distance_m':round(best_d,1),'gauss_weight':float(np.exp(-best_d/SIGMA)),
                         's_lat':best_s['lat'],'s_lon':best_s['lon'],
                         'r_lat':r['lat'],'r_lon':r['lon'],'edge_type':'fallback'})
adj_exp = pd.concat([adj, pd.DataFrame(new_rows)], ignore_index=True)

# ── sewer→sewer 엣지 (동일 배수구역 + 500m 이내) ─────────────────────────────
sewer_geo2 = sewer_geo[sewer_geo['배수구역'].notna()]
ids  = sewer_geo2['sensor_id'].tolist()
lats = sewer_geo2['lat'].tolist()
lons = sewer_geo2['lon'].tolist()
dsts = sewer_geo2['배수구역'].tolist()
ss_edges = []
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        if dsts[i] != dsts[j]: continue
        d = haversine(lats[i],lons[i],lats[j],lons[j])
        if d <= 500:
            ss_edges.append({'src_sewer_id':ids[i],'dst_sewer_id':ids[j],
                             'distance_m':round(d,1),'gauss_weight':float(np.exp(-d/SIGMA)),
                             '배수구역':dsts[i]})
df_ss = pd.DataFrame(ss_edges)

os.makedirs('./dataset/processed/features/', exist_ok=True)
adj_exp.to_parquet('./dataset/processed/features/adjacency_expanded.parquet', index=False)
df_ss.to_parquet('./dataset/processed/features/sewer_sewer_edges.parquet', index=False)
print(f'sewer→road: {len(adj_exp)}개 / sewer→sewer: {len(df_ss)}개')
```

---

### Step 08. 공통 기간 분리 (2024-01 ~ 2025-08)

하수관로(2022~2025)와 도로노면(2024~2026)의 교집합인 2024-01 ~ 2025-08만 추출합니다.

```python
import glob, os, gc
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

OVERLAP_START = pd.Timestamp('2024-01-01')
OVERLAP_END   = pd.Timestamp('2025-08-31 23:59:59')
os.makedirs('./dataset/processed/features/overlap/', exist_ok=True)

def stream_merge(files, out_path, schema, label):
    writer = pq.ParquetWriter(out_path, schema)
    total  = 0
    for f in sorted(files):
        df = pd.read_parquet(f, engine='pyarrow')
        df = df[(df['timestamp']>=OVERLAP_START)&(df['timestamp']<=OVERLAP_END)]
        if df.empty: continue
        for field in schema:
            if field.name not in df.columns: continue
            if pa.types.is_timestamp(field.type):
                df[field.name] = df[field.name].astype('datetime64[ns]')
            elif pa.types.is_floating(field.type):
                df[field.name] = pd.to_numeric(df[field.name], errors='coerce')
            elif pa.types.is_integer(field.type):
                df[field.name] = df[field.name].fillna(0).astype('int8')
        writer.write_table(pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False))
        total += len(df); del df; gc.collect()
    writer.close()
    print(f'{label}: {total:,}행 → {os.path.getsize(out_path)/1e6:.0f} MB')

SEWER_FEAT_SCHEMA = pq.read_schema('./dataset/processed/features/sewer/하수관로_수위_현황_202401.parquet')
ROAD_FEAT_SCHEMA  = pq.read_schema('./dataset/processed/features/road/2024년 1월.parquet')

sewer_files = (sorted(glob.glob('./dataset/processed/features/sewer/하수관로_수위_현황_2024*.parquet'))+
               sorted(glob.glob('./dataset/processed/features/sewer/tv_swm_wal_mea_2025*.parquet')))
road_files  = (sorted(glob.glob('./dataset/processed/features/road/2024년*.parquet'))+
               [f for f in sorted(glob.glob('./dataset/processed/features/road/2025년*.parquet'))
                if any(f'2025년 {m}월' in f for m in ['1','2','3','4','5','6','7','8'])])

stream_merge(sewer_files, './dataset/processed/features/overlap/sewer_overlap.parquet', SEWER_FEAT_SCHEMA, '하수관로')
stream_merge(road_files,  './dataset/processed/features/overlap/road_overlap.parquet',  ROAD_FEAT_SCHEMA,  '도로노면')
```

---

### Step 09. 정규화

**핵심**: Train 기간(2024년) 데이터에서만 파라미터(scale, mean, std)를 계산하고, Val/Test에는 그 파라미터를 그대로 적용합니다. 그렇지 않으면 미래 데이터의 통계가 학습에 누출되는 **data leakage**가 발생합니다.

```python
import pandas as pd, numpy as np, os, gc
import pyarrow as pa, pyarrow.parquet as pq, glob

sewer_node = pd.read_parquet('./dataset/processed/cleaned/sewer_node.parquet', engine='pyarrow')
road_node  = pd.read_parquet('./dataset/processed/cleaned/road_node.parquet',  engine='pyarrow')

# ── Train 기간(2024년) 데이터로만 파라미터 계산 ───────────────────────────────
train_s = pd.concat([pd.read_parquet(f, engine='pyarrow', columns=['sensor_id','level','level_diff'])
                     for f in sorted(glob.glob('./dataset/processed/features/sewer/하수관로_수위_현황_2024*.parquet'))])
s_stats = train_s.groupby('sensor_id').agg(
    level_max  =('level','max'),
    level_p99  =('level', lambda x: x.quantile(0.99)),
    diff_mean  =('level_diff','mean'),
    diff_std   =('level_diff','std')).reset_index()
s_stats['phys_max']    = s_stats['sensor_id'].map(sewer_node.set_index('sensor_id')['max_level_m'].to_dict())
s_stats['level_scale'] = s_stats[['level_max','phys_max']].max(axis=1).fillna(s_stats['level_p99'])
s_stats.loc[s_stats['level_scale']<=0,'level_scale'] = 1.0
s_stats['diff_std'] = s_stats['diff_std'].fillna(1.0).replace(0,1.0)
del train_s

train_r = pd.concat([pd.read_parquet(f, engine='pyarrow', columns=['sensor_id','level','level_diff'])
                     for f in sorted(glob.glob('./dataset/processed/features/road/2024년*.parquet'))])
r_stats = train_r.groupby('sensor_id').agg(
    level_max=('level','max'),
    diff_mean=('level_diff','mean'),
    diff_std =('level_diff','std')).reset_index()
r_stats['phys_max']    = r_stats['sensor_id'].map(road_node.set_index('sensor_id')['max_level_cm'].to_dict())
r_stats['level_scale'] = r_stats[['level_max','phys_max']].max(axis=1).fillna(96.0)
r_stats.loc[r_stats['level_scale']<=0,'level_scale'] = 96.0
r_stats['diff_std'] = r_stats['diff_std'].fillna(1.0).replace(0,1.0)
del train_r

s_stats.to_parquet('./dataset/processed/features/overlap/sewer_norm_params.parquet', index=False)
r_stats.to_parquet('./dataset/processed/features/overlap/road_norm_params.parquet',  index=False)
print(f'파라미터 저장: 하수 {len(s_stats)}개 / 도로 {len(r_stats)}개')

# ── 정규화 적용 ────────────────────────────────────────────────────────────────
def apply_norm(df, scale_map, dmean_map, dstd_map, is_sewer=True):
    df['_sc'] = df['sensor_id'].map(scale_map).fillna(10.0 if is_sewer else 96.0)
    df['_dm'] = df['sensor_id'].map(dmean_map).fillna(0.0)
    df['_ds'] = df['sensor_id'].map(dstd_map).fillna(1.0)
    df['level_norm']      = (df['level']/df['_sc']).clip(0,1).astype('float32')
    df['level_diff_norm'] = ((df['level_diff']-df['_dm'])/df['_ds']).clip(-5,5).astype('float32')
    df.drop(columns=['_sc','_dm','_ds'], inplace=True)
    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24).astype('float32')
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24).astype('float32')
    df['month_sin'] = np.sin(2*np.pi*df['month']/12).astype('float32')
    df['month_cos'] = np.cos(2*np.pi*df['month']/12).astype('float32')
    df['level']      = df['level'].astype('float32')
    df['level_diff'] = df['level_diff'].astype('float32')
    return df

s_params = pd.read_parquet('./dataset/processed/features/overlap/sewer_norm_params.parquet')
r_params = pd.read_parquet('./dataset/processed/features/overlap/road_norm_params.parquet')
s_scale  = s_params.set_index('sensor_id')['level_scale'].to_dict()
s_dmean  = s_params.set_index('sensor_id')['diff_mean'].to_dict()
s_dstd   = s_params.set_index('sensor_id')['diff_std'].to_dict()
r_scale  = r_params.set_index('sensor_id')['level_scale'].to_dict()
r_dmean  = r_params.set_index('sensor_id')['diff_mean'].to_dict()
r_dstd   = r_params.set_index('sensor_id')['diff_std'].to_dict()

for label, src, dst, is_sewer in [
    ('하수관로', './dataset/processed/features/overlap/sewer_overlap.parquet',
              './dataset/processed/features/overlap/sewer_normalized.parquet', True),
    ('도로노면', './dataset/processed/features/overlap/road_overlap.parquet',
              './dataset/processed/features/overlap/road_normalized.parquet',  False)]:
    df = pd.read_parquet(src, engine='pyarrow')
    scale_m, dm, ds = (s_scale,s_dmean,s_dstd) if is_sewer else (r_scale,r_dmean,r_dstd)
    df = apply_norm(df, scale_m, dm, ds, is_sewer)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst)
    print(f'{label}: {len(df):,}행 → {os.path.getsize(dst)/1e6:.0f} MB')
    del df; gc.collect()
```

---

### Step 10. Train / Val / Test 시간 기반 분리

```
Train : 2024-01-01 ~ 2024-12-31  (12개월)
Val   : 2025-01-01 ~ 2025-05-31  (5개월)
Test  : 2025-06-01 ~ 2025-08-31  (3개월, 장마철)
```

> Val/Test에도 Step 09에서 계산한 Train 정규화 파라미터를 그대로 사용합니다.

```python
import pandas as pd, os, gc
import pyarrow as pa, pyarrow.parquet as pq

SPLITS = {
    'train': (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31 23:59:59')),
    'val'  : (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-05-31 23:59:59')),
    'test' : (pd.Timestamp('2025-06-01'), pd.Timestamp('2025-08-31 23:59:59')),
}
BASE = './dataset/processed/features/overlap/'
for s in SPLITS:
    os.makedirs(BASE+s+'/', exist_ok=True)

for label in ['sewer', 'road']:
    src    = BASE+f'{label}_normalized.parquet'
    schema = pq.read_schema(src)
    df     = pd.read_parquet(src, engine='pyarrow')
    print(f'[{label}] {len(df):,}행 분리 중...')
    for split, (s, e) in SPLITS.items():
        out = BASE+f'{split}/{label}_{split}.parquet'
        sub = df[(df['timestamp']>=s)&(df['timestamp']<=e)].reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(sub, schema=schema, preserve_index=False), out)
        print(f'  {split}: {len(sub):,}행 / {os.path.getsize(out)/1e6:.0f} MB')
    del df; gc.collect()
```

---

### Step 11. GNN 설정 파일 생성

노드 인덱스 매핑, 클래스 가중치, feature 목록, 강우 placeholder를 `gnn_config.json`으로 저장합니다.

```python
import pandas as pd, numpy as np, json, os

BASE    = './dataset/processed/features/'
adj_exp = pd.read_parquet(BASE+'adjacency_expanded.parquet')
ss      = pd.read_parquet(BASE+'sewer_sewer_edges.parquet')
r_norm  = pd.read_parquet(BASE+'overlap/road_normalized.parquet',  engine='pyarrow', columns=['sensor_id'])
s_norm  = pd.read_parquet(BASE+'overlap/sewer_normalized.parquet', engine='pyarrow', columns=['sensor_id'])

r_with_data = set(r_norm['sensor_id'].unique())
s_with_data = set(s_norm['sensor_id'].unique())

adj_valid = adj_exp[adj_exp['road_id'].isin(r_with_data)&adj_exp['sewer_id'].isin(s_with_data)].reset_index(drop=True)
ss_valid  = ss[ss['src_sewer_id'].isin(s_with_data)&ss['dst_sewer_id'].isin(s_with_data)].reset_index(drop=True)
adj_valid.to_parquet(BASE+'adjacency_expanded.parquet', index=False)
ss_valid.to_parquet(BASE+'sewer_sewer_edges.parquet', index=False)

sewer_ids = sorted((set(adj_valid['sewer_id'])|set(ss_valid['src_sewer_id']))&s_with_data)
road_ids  = sorted(set(adj_valid['road_id'])&r_with_data)
pd.DataFrame({'sensor_id':sewer_ids,'node_idx':range(len(sewer_ids)),'node_type':'sewer'}).to_parquet(BASE+'sewer_node_index.parquet',index=False)
pd.DataFrame({'sensor_id':road_ids, 'node_idx':range(len(road_ids)), 'node_type':'road'}).to_parquet(BASE+'road_node_index.parquet',index=False)

# 클래스 가중치 (Train 기준)
r_train = pd.read_parquet(BASE+'overlap/train/road_train.parquet', engine='pyarrow', columns=['flood_flag','flood_stage'])
n_pos   = int((r_train['flood_flag']==1).sum())
n_neg   = int((r_train['flood_flag']==0).sum())
stage_w = {int(k): round(len(r_train)/(5*v),2) for k,v in r_train['flood_stage'].value_counts().items()}

config = {
    'graph': {
        'sewer_road_edges':  len(adj_valid),
        'sewer_sewer_edges': len(ss_valid),
        'sewer_nodes': len(sewer_ids),
        'road_nodes':  len(road_ids),
        'sigma_m': 300, 'threshold': 0.1
    },
    'node_features': {
        'sewer': ['level_norm','level_diff_norm','fill_rate',
                  'hour_sin','hour_cos','month_sin','month_cos','season','is_weekend'],
        'road':  ['level_norm','level_diff_norm','flood_flag','flood_stage',
                  'hour_sin','hour_cos','month_sin','month_cos','season','is_weekend']
    },
    'target_features': {
        'road_regression':    'level_norm',
        'road_classification':'flood_flag',
        'road_multiclass':    'flood_stage'
    },
    'class_weights': {
        'binary': {'pos_weight':round(n_neg/n_pos,2),'n_positive':n_pos,'n_negative':n_neg,
                   'imbalance_ratio':f'1:{round(n_neg/n_pos)}'},
        'multiclass_stage': stage_w
    },
    'temporal': {'input_steps':6,'output_steps':3,'resolution_min':10},
    'train_val_test': {
        'train':['2024-01-01','2024-12-31'],
        'val':  ['2025-01-01','2025-05-31'],
        'test': ['2025-06-01','2025-08-31']
    },
    'future_extension': {
        'rainfall': {
            'feature_name': 'rainfall_norm',
            'description':  '시간당 강우량 (mm/hr), 최근접 AWS 기상 관측소',
            'status':       'placeholder',
            'position_in_sewer_features': 9,
            'note':         'AWS 데이터 확보 후 sewer_normalized.parquet 재생성 필요'
        }
    }
}

with open(BASE+'gnn_config.json','w',encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print('=== GNN 데이터셋 준비 완료 ===')
print(f'하수관 노드: {len(sewer_ids)}개 / 도로 노드: {len(road_ids)}개')
print(f'sewer→road: {len(adj_valid)}개 / sewer→sewer: {len(ss_valid)}개')
print(f'클래스 불균형: 1:{round(n_neg/n_pos)}')
```

---

## 5. 최종 산출물

| 파일 | 설명 | 크기 |
|------|------|------|
| `raw_parquet/sewer/` | 변환된 raw Parquet (하수관로, 44개) | 1.5 GB |
| `raw_parquet/road/` | 변환된 raw Parquet (도로, 24개) | 394 MB |
| `cleaned/sewer_cleaned.parquet` | 이상치 제거 + 보간 완료 | — |
| `cleaned/road_cleaned.parquet` | 이상치 제거 + 보간 완료 | — |
| `cleaned/sewer_node.parquet` | 하수관로 노드 메타데이터 (456개) | — |
| `cleaned/road_node.parquet` | 도로노면 노드 메타데이터 (112개) | — |
| `features/adjacency_expanded.parquet` | sewer→road 엣지 (501개) | — |
| `features/sewer_sewer_edges.parquet` | sewer→sewer 엣지 (1,192개) | — |
| `features/overlap/sewer_normalized.parquet` | 정규화된 하수관로 전체 | 900 MB |
| `features/overlap/road_normalized.parquet` | 정규화된 도로노면 전체 | 319 MB |
| `features/overlap/train/` | Train 분리 결과 | 659 MB |
| `features/overlap/val/` | Val 분리 결과 | 297 MB |
| `features/overlap/test/` | Test 분리 결과 (장마철) | 257 MB |
| `features/gnn_config.json` | GNN 그래프 설정 및 메타데이터 | — |

## 6. 데이터 로드 방법

```python
import pandas as pd

# 특정 split 로드
sewer_train = pd.read_parquet('./dataset/processed/features/overlap/train/sewer_train.parquet')
road_train  = pd.read_parquet('./dataset/processed/features/overlap/train/road_train.parquet')

# 그래프 설정 확인
import json
with open('./dataset/processed/features/gnn_config.json', encoding='utf-8') as f:
    config = json.load(f)

print(config['graph'])
# {'sewer_road_edges': 501, 'sewer_sewer_edges': 1192,
#  'sewer_nodes': 456, 'road_nodes': 112, 'sigma_m': 300, 'threshold': 0.1}
```
