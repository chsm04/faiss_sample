from sentence_transformers import SentenceTransformer

# 임베딩 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트 리스트
texts = ["Hello world", "How are you?", "FAISS is great for vector search"]

# 텍스트를 벡터로 변환
vectors = model.encode(texts)
print(vectors.shape)  # (3, 384) - 3개의 텍스트, 384 차원의 벡터



import faiss
import numpy as np

# 벡터 데이터를 NumPy 배열로 변환 (FAISS는 NumPy 배열을 사용함)
vector_data = np.array(vectors).astype('float32')

# 벡터의 차원 수 확인
d = vector_data.shape[1]

# 인덱스 생성 (L2 거리 기반 인덱스)
index = faiss.IndexFlatL2(d)

# 벡터 데이터 인덱스에 추가
index.add(vector_data)

# 인덱스에 추가된 벡터의 수 확인
print(f"Total vectors in the index: {index.ntotal}")