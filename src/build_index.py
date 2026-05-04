from pathlib import Path
from sentence_transformers import SentenceTransformer

from utils import load_text, split_into_chunks, save_pickle


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "knowledge.txt"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    text = load_text(data_path)
    chunks = split_into_chunks(text)
    #转化成语义向量
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

    save_pickle(chunks, output_dir / "chunks.pkl")
    save_pickle(chunk_embeddings, output_dir / "chunk_embeddings.pkl")

    print("Embedding 索引构建完成")
    print(f"共生成 {len(chunks)} 个文本块")
    print(f"Embedding shape: {chunk_embeddings.shape}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(chunk)


if __name__ == "__main__":
    main()