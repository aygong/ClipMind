import cv2
from IPython.display import display, HTML
import json
import networkx as nx
import numpy as np
import os
import pickle
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm


def read_file_ids(trace_dir, overwrite=False):
    """
    Generate a list of file IDs from the trace directory.
    """
    pickle_file = os.path.join(trace_dir, "file_ids.pickle")

    if os.path.isfile(pickle_file) and not overwrite:
        # Load existing file IDs
        with open(pickle_file, "rb") as file:
            file_ids = pickle.load(file)
        print(f"✅ Loaded a cached pickle file from `{pickle_file}`")
    else:
        # Generate file IDs from video files
        video_dir = os.path.join(trace_dir, "videos")
        file_ids = []

        for file_name in os.listdir(video_dir):
            video_path = os.path.join(video_dir, file_name)
            try:
                video = cv2.VideoCapture(video_path)
                if not video.isOpened():
                    raise ValueError(f"Unable to open video: {file_name}")
            except cv2.error as error:
                print(f"[OpenCV Error] {file_name}: {error}")
            except Exception as error:
                print(f"[Exception] {file_name}: {error}")
            else:
                base_name = os.path.splitext(file_name)[0]
                file_id = re.sub(r"[^0-9]", "", base_name)
                file_ids.append(file_id)

        # Save file IDs to pickle
        with open(pickle_file, "wb") as file:
            pickle.dump(file_ids, file)
        print(f"✅ Saved a pickle file to `{pickle_file}`")

    return file_ids


def load_pickle_file(file_path):
    """
    Load a pickle file from the specified path.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ File is not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded: {file_path}")
    
    return data


def build_file_embeddings(trace_dir, file_ids, features_embedding_dirs, output_path):
    """
    Build or load file embeddings from feature directories.
    """
    if os.path.isfile(output_path):
        # Load existing file embeddings
        with open(output_path, "rb") as f:
            file_embeddings = pickle.load(f)
        print(f"✅ Loaded cached file embeddings from `{output_path}`")
    else:
        print("🔍 Features:", [feature for feature, _ in features_embedding_dirs])
        file_embeddings = {}

        for file_id in tqdm(file_ids, desc="Reading embeddings"):
            file_embeddings[file_id] = {}

            for feature, embedding_dir in features_embedding_dirs:
                embedding_path = os.path.join(trace_dir, embedding_dir, f"{file_id}.npy")

                if os.path.isfile(embedding_path):
                    file_embeddings[file_id][feature] = np.load(embedding_path)
                else:
                    raise FileNotFoundError(f"❌ Missing embedding file: {embedding_path}")

        # Save file embeddings to pickle
        with open(output_path, "wb") as f:
            pickle.dump(file_embeddings, f)
        print(f"✅ Saved file embeddings to `{output_path}`")

    # Save the shapes of embeddings
    shapes = {}
    file_embedding = next(iter(file_embeddings.values()))

    for feature, embedding in file_embedding.items():
        shapes[feature] = embedding.reshape(1, -1).shape[1]

    print("📐 Embedding dimensions by feature:")
    for source, dim in shapes.items():
        print(f" - {source}: {dim}")

    return file_embeddings, shapes


def wrapped_print(text, words_per_line=15):
    """
    Print long text in chunks of a specified number of words per line.
    """
    words = text.split()
    for i in range(0, len(words), words_per_line):
        print(" ".join(words[i:i + words_per_line]))


def valid_input(valid_options, prompt="Your input: "):
    """
    Prompt the user until a valid input is provided.
    """
    valid_options = [option.lower() for option in valid_options]

    while True:
        response = input(prompt).strip().lower()
        
        if response in valid_options:
            return response
        
        print(f"❌ Invalid input. Please enter one of: {', '.join(valid_options)}.")


def get_next_unannotated_pair(pairs, annotations):
    """
    Get next unannotated video pair
    """
    for pair in pairs:
        if pair not in annotations:
            return pair
    return None, None


def display_metadata_and_videos(video_ids, metadata, trace_dir, print_width=40):
    """
    Display metadata and corresponding videos for a list of video IDs.
    """
    print("-" * print_width)
    for idx, file_id in enumerate(video_ids, start=1):
        print(f"\u26D3 \033[1m{file_id} (Video {idx})\033[0m \u26D3")
        wrapped_print(metadata[file_id]["desc"])
        print("-" * print_width)

    # Generate HTML for videos
    video_html = '<div style="display: flex; gap: 20px;">'
    
    for idx, file_id in enumerate(video_ids, start=1):
        title = f"Video {idx}"
        video_path = os.path.join(trace_dir, "videos", f"{file_id}.mp4")
        video_html += f"""
        <div>
            <h4>{title}</h4>
            <video width="160" autoplay loop controls>
                <source src="{video_path}" type="video/mp4">
            </video>
        </div>
        """
    
    video_html += '</div>'
    
    display(HTML(video_html))


def compute_cosine_similarity(annotation, combo, file_embeddings):
    """
    Compute cosine similarity for all annotated video pairs using the specified feature combination.
    """
    nof_pairs = len(annotation)
    similarity = np.zeros(nof_pairs)
    ground_truth = np.zeros(nof_pairs, dtype=bool)

    for idx, (id_1, id_2) in enumerate(annotation):
        v1, v2 = np.zeros((1, 0)), np.zeros((1, 0))
        
        for feature in combo:
            e1 = file_embeddings[id_1][feature].reshape(1, -1)
            e2 = file_embeddings[id_2][feature].reshape(1, -1)
            v1 = np.concatenate((v1, e1 / np.linalg.norm(e1)), axis=1)
            v2 = np.concatenate((v2, e2 / np.linalg.norm(e2)), axis=1)
        
        similarity[idx] = cosine_similarity(v1, v2)[0, 0]
        ground_truth[idx] = annotation[(id_1, id_2)]

    return similarity, ground_truth


def compute_metrics_across_thresholds(similarity, ground_truth, thresholds):
    """
    Evaluate precision, recall, F1 score, and accuracy across a range of similarity thresholds.
    """
    thresholded_outputs = [similarity > t for t in thresholds]

    scores = [
        precision_recall_fscore_support(
            ground_truth, decision, average='binary', zero_division=0
        )[:3] + (accuracy_score(ground_truth, decision),)
        for decision in thresholded_outputs
    ]

    precision, recall, f1, accuracy = zip(*scores)

    return accuracy, precision, recall, f1


def build_ordered_embedding_matrix(trace_dir, file_ids, file_embeddings, shapes):
    """
    Build a concatenated embedding matrix for ordered file IDs based on viewing history.
    """
    # Load viewing history
    viewing_path = os.path.join(trace_dir, "viewing.json")
    with open(viewing_path, "r", encoding="utf-8") as f:
        viewing_history = json.load(f)

    # Extract ordered file IDs based on viewing history
    ordered_ids = []
    for _, file_id in viewing_history:
        if file_id in file_ids and file_id not in ordered_ids:
            ordered_ids.append(file_id)

    print(f"🗂️ Ordered {len(ordered_ids)} out of {len(file_ids)} available file IDs")

    # Construct concatenated embedding matrix
    concat_dim = sum(shapes[feature] for feature in shapes)
    concat_embedding_matrix = np.zeros((len(ordered_ids), concat_dim))

    for idx, file_id in enumerate(tqdm(ordered_ids, desc="Embedding files")):
        concat_embedding = []

        for feature in shapes:
            embedding = file_embeddings[file_id][feature].reshape(1, -1)
            normalized_embedding = embedding / np.linalg.norm(embedding, ord=2)
            concat_embedding.append(normalized_embedding)

        concat_embedding_matrix[idx, :] = np.concatenate(concat_embedding, axis=1)

    print(f"✅ Completed matrix construction. Shape: {concat_embedding_matrix.shape}")
    
    return concat_embedding_matrix, ordered_ids


def analyze_temporal_windows(ordered_ids, keywords, binary_similarity, window_lengths=[10], step=1):
    """
    Analyze video similarity patterns using sliding windows and graph-based clustering.
    """
    start_normalized_orders = {}
    start_components_keywords = {}
    
    for length in window_lengths:
        start_normalized_orders[length] = []
        start_components_keywords[length] = []
        
        window_starts = range(0, len(binary_similarity) - length + 1, step)

        for start in window_starts:
            # Build graph for current window
            window_nodes = range(start, start + length)
            G = nx.Graph()
            G.add_nodes_from(window_nodes)

            for i in window_nodes:
                for j in range(i + 1, start + length):
                    if binary_similarity[i, j]:
                        G.add_edge(i, j)

            components = list(nx.connected_components(G))
            largest_order = len(max(components, key=len))
            largest_components = [C for C in components if len(C) == largest_order]

            start_normalized_orders[length].append(largest_order / length)
            
            # Extract component keywords in largest components
            components_keywords = []
            
            if largest_order > 1:
                for component in largest_components:
                    # Count keywords in the current component
                    counts = {}
                    for node in component:
                        keywords_list = keywords[ordered_ids[node]].split(", ")
                        for kw in keywords_list:
                            counts[kw] = counts.get(kw, 0) + 1
                    # Append the current component start and its keywords
                    components_keywords.append((
                        component, [kw for kw, count in counts.items() if count == largest_order]
                    ))
            
            if components_keywords:
                start_components_keywords[length].append([start, components_keywords])
            else:
                start_components_keywords[length].append([start, [({}, [])]])
    
    return start_normalized_orders, start_components_keywords