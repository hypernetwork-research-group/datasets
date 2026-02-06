import os
import fastjsonschema
import json
import requests
import zstandard as zstd
import pandas as pd
import hypernetx as hnx
import ast
import pickle
from collections import defaultdict



def validate_hif_json(filename):
    url = "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json"
    try:
        schema = requests.get(url, timeout=10).json()
    except (requests.RequestException, requests.Timeout):
        with open("../schema/hif_schema.json", "r") as f:
            schema = json.load(f)
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(filename, "r"))
    try:
        validator(hiftext)
        return True
    except Exception:
        return False


def compress_to_zst(filename):
    json_filename = filename.rsplit(".", 1)[0] + ".json"
    zst_filename = json_filename + ".zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(json_filename, "rb") as input_f, open(zst_filename, "wb") as output_f:
        cctx.copy_stream(input_f, output_f)
    return zst_filename


def decompress_from_zst(zst_filename):
    json_filename = zst_filename.rsplit(".", 2)[0] + ".json"
    dctx = zstd.ZstdDecompressor()
    with open(zst_filename, "rb") as input_f, open(json_filename, "wb") as output_f:
        dctx.copy_stream(input_f, output_f)
    return json_filename


def validate_json_syntax(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (ValueError, FileNotFoundError) as e:
        print(f"DEBUG: JSON data contains an error: {e}")
        return False


# https://annualized-gender-data-uspto.s3.amazonaws.com/2023.csv
def convert_patent_to_hif():
    def extract_inventor_mappings(
        df: pd.DataFrame, inventor_cols: list[str]
    ) -> dict[str, list[str]]:
        mapping = {}
        for _, row in df.iterrows():
            patent_id = str(row["patent_number"])
            inventors = [row[col] for col in inventor_cols if pd.notna(row[col])]
            if len(inventors) >= 2:
                mapping[patent_id] = inventors
        return mapping

    def invert_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        inv_map = defaultdict(list)
        for patent, inventors in mapping.items():
            for inv in inventors:
                inv_map[inv].append(patent)
        return dict(inv_map)
    
    path = "path/to/2023.csv"
    path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_csv(path, dtype=str)
    inventor_cols = [f"inventor_name{i}" for i in range(1, 10)]
    patent_to_inventors = extract_inventor_mappings(df, inventor_cols)
    patent_to_inventors = {
        idx: tuple(inventors)
        for idx, (patent, inventors) in enumerate(patent_to_inventors.items())
    }
    rows = []
    for edge_id, inventors in patent_to_inventors.items():
        for inventor in inventors:
            rows.append({"edges": edge_id, "nodes": inventor})
    hg_df = pd.DataFrame(rows)
    H = hnx.Hypergraph(hg_df)
    hnx.to_hif(H, "path/to/patent.json")
    compress_to_zst("path/to/patent.json")


# https://www.kaggle.com/datasets/elvinrustam/coursera-dataset
def convert_coursera_to_hif():
    path = "Coursera.csv"
    path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_csv(path, dtype=str)
    course_to_instructors = defaultdict(list)
    df = df.dropna(subset=["Instructor"])
    for _, row in df.iterrows():
        course = row["Course Title"]
        instructors_raw = row["Instructor"]
        try:
            instructors_raw = instructors_raw.replace(" and ", ",")
        except AttributeError:
            print(
                f"Skipping course {course} due to invalid instructor format.\n {instructors_raw}"
            )
        instructors = [inst.strip() for inst in instructors_raw.split(",")]
        course_to_instructors[course].append(instructors)

    coursera_dict = {
        idx: tuple(set([inst for sublist in instructors for inst in sublist]))
        for idx, (course, instructors) in enumerate(course_to_instructors.items())
    }
    rows = []
    for edge_id, instructors in coursera_dict.items():
        for instructor in instructors:
            rows.append({"edges": edge_id, "nodes": instructor})
    hg_df = pd.DataFrame(rows)
    H = hnx.Hypergraph(hg_df)
    hnx.to_hif(H, "path/to/coursera.json")
    compress_to_zst("path/to/coursera.json")


# https://www.kaggle.com/datasets/payamamanat/imbd-dataset
def convert_imdb_to_hif():
    path = "IMBD.csv"
    path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_csv(path, dtype=str)
    movie_to_actors = defaultdict(list)
    df["stars"] = (
        df["stars"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "")
        .str.split(", ")
        .apply(
            lambda x: [
                actor.replace("Stars:", "").split("|")[0].strip()
                for actor in x
                if actor.strip()
            ]
        )
    )

    df["stars"] = df["stars"].apply(lambda x: [actor for actor in x if actor])

    df["stars"] = df["stars"].apply(
        lambda x: [actor.encode("utf-8", "ignore").decode("utf-8") for actor in x]
    )

    for _, row in df.iterrows():
        movie = row["title"]
        actors = row["stars"]
        movie_to_actors[movie].append(actors)

    imdb_dict = {
        idx: tuple(set([actor for sublist in actors for actor in sublist]))
        for idx, (movie, actors) in enumerate(movie_to_actors.items())
    }

    rows = []
    for edge_id, actors in imdb_dict.items():
        for actor in actors:
            rows.append({"edges": edge_id, "nodes": actor})
    hg_df = pd.DataFrame(rows)
    H = hnx.Hypergraph(hg_df)
    hnx.to_hif(H, "path/to/imdb.json")
    validate_json_syntax("path/to/imdb.json")
    validate_hif_json("path/to/imdb.json")
    with open("path/to/imdb.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("path/to/imdb.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ": "))
    compress_to_zst("path/to/imdb.json")


# https://www.kaggle.com/datasets/Cornell-University/arxiv/
def convert_arxiv_to_hif():
    rows = []
    cctx = zstd.ZstdDecompressor()
    with (
        open(
            "path/to/rows.txt.zst",
            "rb",
        ) as input_f,
        open("rows.list", "wb") as output_f,
    ):
        cctx.copy_stream(input_f, output_f)
    with open("rows.list", "r") as f:
        for line in f:
            # {'edges': 0, 'nodes': 'C. Balázs'}

            row_dict = ast.literal_eval(line.strip())
            rows.append(row_dict)

            # paper_id = paper["id"]
            # authors = paper["authors"].split(", ")
            # for author in authors:
            #     rows.append({"edges": paper_id, "nodes": author})

    hg_df = pd.DataFrame(rows)
    H = hnx.Hypergraph(hg_df)
    hnx.to_hif(H, "path/to/arxiv.json")
    compress_to_zst("path/to/arxiv.json")


# https://graphsandnetworks.com/the-cora-dataset/
def convert_cora_to_hif():
    file = "cora/cora.cites"
    path = os.path.join(os.path.dirname(__file__), file)
    edgelist = pd.read_csv(path, sep="\t", header=None, names=["target", "source"])
    edgelist["label"] = "cites"

    file = "cora/cora.content"
    path = os.path.join(os.path.dirname(__file__), file)
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = ["paper_id"] + feature_names + ["subject"]
    node_data = pd.read_csv(path, sep="\t", header=None, names=column_names)

    paper_to_citations = edgelist.groupby("target")["source"].apply(list).to_dict()

    rows = []
    for edge_id, (paper, citations) in enumerate(paper_to_citations.items()):
        for cited_paper in citations:
            rows.append({"edges": edge_id, "nodes": cited_paper})

    hg_df = pd.DataFrame(rows)

    node_attrs = {}
    for _, row in node_data.iterrows():
        paper_id = row["paper_id"]
        attrs = {
            "subject": row["subject"],
            "features": row[feature_names].values.tolist(),
        }
        node_attrs[paper_id] = attrs

    H = hnx.Hypergraph(hg_df, node_properties=node_attrs)
    hnx.to_hif(H, "path/to/cora.json")
    compress_to_zst("path/to/cora.json")

# https://github.com/malllabiisc/HyperGCN/tree/master/data/cocitation/pubmed
def convert_pubmed_to_hif():
    hypergraph = pickle.load(open("../pubmed/hypergraph.pickle", "rb"))
    features = pickle.load(open("../pubmed/features.pickle", "rb")).todense()
    rows = []
    for edge_id, (paper_id, cited_papers) in enumerate(hypergraph.items()):
        for cited_paper in cited_papers:
            rows.append({"edges": edge_id, "nodes": cited_paper})
    
    hg_df = pd.DataFrame(rows)
    node_attrs = {}
    for node_id in range(features.shape[0]):
        node_attrs[str(node_id)] = {
            "features": features[node_id].tolist()
        }
    H = hnx.Hypergraph(hg_df, node_properties=node_attrs)
    hnx.to_hif(H, "../pubmed/pubmed.json")
    compress_to_zst("../pubmed/pubmed.json")

if __name__ == "__main__":
    # convert_patent_to_hif()
    # convert_coursera_to_hif()
    # convert_imdb_to_hif()
    # convert_arxiv_to_hif() #TODO fix
    # convert_cora_to_hif()
    # convert_pubmed_to_hif()
    print("Done")
