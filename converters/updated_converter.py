import os
import pickle
import networkx as nx
import pandas as pd
import hypernetx as hnx
import zstandard as zstd

def compress_to_zst(filename):
    json_filename = filename.rsplit(".", 1)[0] + ".json"
    zst_filename = json_filename + ".zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(json_filename, "rb") as input_f, open(zst_filename, "wb") as output_f:
        cctx.copy_stream(input_f, output_f)
    return zst_filename

def prettify_json(json_filename):
    with open(json_filename, "r") as f:
        data = f.read()
    data = data.replace("}, {", "},\n{")
    with open(json_filename, "w") as f:
        f.write(data)

def cora_converter():
    path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(path, "cora")
    output_dir = os.path.join(data_dir, "output")
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
    edgelist["label"] = "cites"

    file = os.path.join(data_dir, "cora.content")
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
            "label": row["subject"],
            # "features": row[feature_names].values.tolist(),
            "attrs": {f"{w}": row[w] for w in feature_names}
        }
        node_attrs[paper_id] = attrs

    H = hnx.Hypergraph(hg_df, node_properties=node_attrs)
    print(H)
    hnx.to_hif(H, os.path.join(output_dir, "cora.json"))
    compress_to_zst(os.path.join(output_dir, "cora.json"))
    prettify_json(os.path.join(output_dir, "cora.json"))

# https://github.com/malllabiisc/HyperGCN/tree/master/data/cocitation/pubmed
def convert_pubmed_to_hif():
    path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(path, "pubmed")
    output_dir = os.path.join(data_dir, "output")
    hypergraph = pickle.load(open(os.path.join(data_dir, "hypergraph.pickle"), "rb"))
    features = pickle.load(open(os.path.join(data_dir, "features.pickle"), "rb")).toarray()
    labels = pickle.load(open(os.path.join(data_dir, "labels.pickle"), "rb"))
    # convert labes from np.int64 to int
    labels = [int(label) for label in labels]
    rows = []
    for edge_id, (paper_id, cited_papers) in enumerate(hypergraph.items()):
        for cited_paper in cited_papers:
            rows.append({"edges": edge_id, "nodes": cited_paper})
    
    hg_df = pd.DataFrame(rows)
    node_attrs = {}
    for node_id in range(features.shape[0]):
        attrs = {
            "attrs": {f"f{ii}": float(features[node_id][ii]) for ii in range(len(features[node_id]))},
            "label": labels[node_id]
        }
        node_attrs[node_id] = attrs

    H = hnx.Hypergraph(hg_df, node_properties=node_attrs)
    hnx.to_hif(H, os.path.join(output_dir, "pubmed.json"))
    compress_to_zst(os.path.join(output_dir, "pubmed.json"))
    prettify_json(os.path.join(output_dir, "pubmed.json"))

if __name__ == "__main__":
    # cora_converter()
    convert_pubmed_to_hif()
