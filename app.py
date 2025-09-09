from flask import Flask, render_template, request
from RecommendationEngine import RecommendationEngine

app = Flask(__name__)

# Load engine once at startup
engine = RecommendationEngine(
    faiss_abstract_idx="data/index_abstract.faiss",
    faiss_title_idx="data/index_title.faiss",
    paper_metadata="data/papers.csv"
)

@app.route("/", methods=["GET"])
def home():
    paper_estimate = engine.get_total_papers()
    return render_template("home.html", num_papers = paper_estimate)

@app.route("/search", methods=["GET", "POST"])
def search():
    return render_template("search.html")

@app.route("/results", methods=["POST"])
def results():
    query_title = request.form["title"]
    k = int(request.form["k"])
    recommendations = engine.recommend_from_abstract(query_title, k)
    return render_template("results.html", query=query_title, recommendations=recommendations)

if __name__ == "__main__":
    app.run()
