import os
import requests
import faiss
import json
import torch
import tqdm
import numpy as np
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

corpus_names = {
    "PubMed": ["pubmed"]
}

retriever_names = {
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"]
}

class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]

class Retriever: 

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="pubmed", db_dir="./corpus", HNSW=False, **kwarg):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
        
        self.metadatas = {}  # Updated to dictionary to match expected format

        print(f"Initializing Retriever for corpus: {self.corpus_name}, retriever: {self.retriever_name}")
        
        # Automatically download required files if they are missing
        self.download_required_files()

        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            print(f"FAISS index found at {self.index_dir}. Attempting to load the index and metadata...")
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.load_metadata()
        else:
            # If no index exists, create one using the pre-computed embeddings
            print(f"No FAISS index found at {self.index_dir}. Proceeding to load pre-computed embeddings...")
            h_dim = self.load_precomputed_embeddings()
            print(f"[In progress] Embedding finished! The dimension of the embeddings is {h_dim}.")
            self.index = self.construct_index(h_dim, HNSW=HNSW)
            print("[Finished] Corpus indexing finished!")           

        self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_function.eval()

    def download_required_files(self):
        """Downloads required files for the Retriever if they do not already exist."""
        # List of base filenames to download for embedding and metadata files
        embed_files = [f"embeds_chunk_{i}.npy" for i in range(38)]  # Embedding files: 0 to 3
        pmid_files = [f"pmids_chunk_{i}.json" for i in range(38)]  # PMIDs metadata files: 0 to 3
        pubmed_files = [f"pubmed_chunk_{i}.json" for i in range(38)]  # PubMed metadata files: 0 to 3

        # Combine all files to create a single list
        files_to_download = embed_files + pmid_files + pubmed_files

        base_url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/"
        os.makedirs(self.chunk_dir, exist_ok=True)

        # Loop through the list of files and download them if they don't exist
        for filename in files_to_download:
            url = base_url + filename
            local_filename = os.path.join(self.chunk_dir, filename)
            if not os.path.exists(local_filename):
                print(f"Downloading {url} to {local_filename}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 8192  # 8 KB per chunk
                    with open(local_filename, 'wb') as f, tqdm.tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=block_size):
                            f.write(chunk)
                            bar.update(len(chunk))
                print(f"Downloaded {local_filename}")
            else:
                print(f"File {local_filename} already exists. Skipping download.")
                
    def load_precomputed_embeddings(self):
        """Load pre-computed embeddings from the provided files using memory mapping to reduce memory usage."""
        embedding_files = sorted([f for f in os.listdir(self.chunk_dir) if f.startswith("embeds_chunk_") and f.endswith(".npy")])
        metadata_files = sorted([f for f in os.listdir(self.chunk_dir) if f.startswith("pubmed_chunk_") and f.endswith(".json")])

        if not embedding_files:
            raise FileNotFoundError("No embedding files found in the specified directory.")
        if not metadata_files:
            raise FileNotFoundError("No metadata files found in the specified directory.")

        embeddings = []
        self.metadatas = {}  # Dictionary to store metadata by unique ID

        print(f"Found {len(embedding_files)} embedding files and {len(metadata_files)} metadata files.")

        for embed_file, metadata_file in zip(embedding_files, metadata_files):
            embed_path = os.path.join(self.chunk_dir, embed_file)
            metadata_path = os.path.join(self.chunk_dir, metadata_file)

            print(f"Loading embeddings from: {embed_path}")
            if not os.path.exists(embed_path):
                print(f"Warning: Embedding file not found: {embed_path}")
                continue

            print(f"Loading metadata from: {metadata_path}")
            if not os.path.exists(metadata_path):
                print(f"Warning: Metadata file not found: {metadata_path}")
                continue

            # Use memory-mapped numpy array to load embeddings in chunks to reduce memory usage
            curr_embeddings = np.load(embed_path, mmap_mode='r')
            embeddings.append(curr_embeddings)
            print(f"Loaded {curr_embeddings.shape[0]} embeddings from {embed_file}")

            # Load the metadata in dictionary format
            try:
                with open(metadata_path, 'r') as f:
                    curr_metadata = json.load(f)
                    if isinstance(curr_metadata, dict):
                        self.metadatas.update(curr_metadata)  # Store metadata by unique ID
                        print(f"Successfully loaded {len(curr_metadata)} metadata entries from {metadata_file}")
                    else:
                        print(f"Warning: Metadata file {metadata_file} does not contain a dictionary of metadata.")
            except json.JSONDecodeError as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
                continue

        if not self.metadatas:
            print("Error: Metadata dictionary is empty after loading.")
            raise ValueError("Metadata is empty after loading. Please ensure that the metadata files are correctly formatted and contain data.")

        print(f"Total number of embedding files loaded: {len(embeddings)}")
        return curr_embeddings.shape[-1]  # Assuming all embeddings have the same dimensionality


    def construct_index(self, h_dim=768, HNSW=False, M=32, batch_size=50000):
        """Constructs a FAISS index using the loaded embeddings and converts metadata to JSONL format."""

        # Ensure the output directory for metadata exists
        os.makedirs(self.index_dir, exist_ok=True)
        metadata_path = os.path.join(self.index_dir, "metadatas.jsonl")

        if HNSW:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            index = faiss.IndexFlatIP(h_dim)
            
        print("[In progress] Building FAISS index with pre-computed embeddings...")

        embedding_files = sorted([f for f in os.listdir(self.chunk_dir) if f.startswith("embeds_chunk_") and f.endswith(".npy")])
        metadata_files = sorted([f for f in os.listdir(self.chunk_dir) if f.startswith("pubmed_chunk_") and f.endswith(".json")])

        with open(metadata_path, 'w') as f_out:  # Open the metadata JSONL file for writing
            total_metadata_count = 0  # To keep track of the number of metadata entries
            for embed_file, meta_file in tqdm.tqdm(zip(embedding_files, metadata_files), total=len(embedding_files)):
                embed_path = os.path.join(self.chunk_dir, embed_file)

                # Use memory-mapped numpy array to load embeddings in chunks
                curr_embed = np.load(embed_path, mmap_mode='r').astype(np.float32)

                # Process embeddings in smaller batches
                for start_idx in range(0, curr_embed.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, curr_embed.shape[0])
                    index.add(curr_embed[start_idx:end_idx])

                # Load the metadata for the current chunk
                with open(os.path.join(self.chunk_dir, meta_file), 'r') as f:
                    metadata_entries = json.load(f)

                if not metadata_entries:
                    print(f"Warning: Metadata file {meta_file} is empty or not properly formatted.")

                # Convert metadata to JSONL format and write to the file
                for key, entry in metadata_entries.items():
                    json_line = json.dumps({'index': key, 'source': meta_file, **entry})
                    f_out.write(json_line + '\n')
                    total_metadata_count += 1

            print(f"[Finished] Indexing complete and FAISS index saved. Total metadata entries written: {total_metadata_count}")

        faiss.write_index(index, os.path.join(self.index_dir, "faiss.index"))
        return index

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        assert type(question) == str
        question = [question]

        if not self.metadatas:
            raise ValueError("Metadata is empty. Ensure that metadata is properly loaded before retrieval.")

        with torch.no_grad():
            query_embed = self.embedding_function.encode(question, **kwarg)
        res_ = self.index.search(query_embed, k=k)
        
        # Use unique IDs to fetch metadata from the dictionary
        ids = [str(i) for i in res_[1][0]]
        indices = [self.metadatas[id] for id in ids if id in self.metadatas]

        scores = res_[0][0].tolist()
        
        if id_only:
            return [{"id": id} for id in ids], scores
        else:
            return self.idx2txt(indices), scores

    def idx2txt(self, indices):
        '''
        Input: List of Dict( {"d": date, "t": title, "a": abstract } )
        Output: List of str
        '''
        return [{"title": item["t"], "content": item["a"]} for item in indices]

    
    def load_metadata(self):
        """Load metadata information from the existing index directory."""
        metadata_file = os.path.join(self.index_dir, "metadatas.jsonl")
        if os.path.exists(metadata_file):
            print(f"Loading metadata from {metadata_file}...")
            with open(metadata_file, 'r') as f:
                self.metadatas = {entry['index']: entry for entry in (json.loads(line) for line in f.read().strip().split('\n'))}
            print(f"Loaded {len(self.metadatas)} metadata entries from the file.")
        else:
            print(f"Metadata file {metadata_file} not found.")

class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="PubMed", db_dir="./corpus", HNSW=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str

        texts = []
        scores = []

        k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)

        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        '''
            Merge the texts and scores from different retrievers
        '''
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]

            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]

            texts[i] = [texts_all[idx] for idx in sorted_index]
            scores[i] = [scores_all[idx] for idx in sorted_index]

            for j, item in enumerate(texts[i]):
                # Use 'index' as the unique identifier instead of 'id'
                item_id = item.get("index")
                if item_id in RRF_dict:
                    RRF_dict[item_id]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item_id]["count"] += 1
                else:
                    RRF_dict[item_id] = {
                        "index": item_id,
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                    }

        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)

        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [dict((key, item[1][key]) for key in ("index", "title", "content")) for item in RRF_list[:k]]
            scores = [item[1]["score"] for item in RRF_list[:k]]

        return texts, scores
