import flask
from requests import HTTPError

bp = flask.Blueprint('main', __name__)

@bp.route("/", methods=["GET"])
def home():
    paper_estimate = flask.current_app.papers.get_total_papers()
    return flask.render_template("home.html", num_papers = paper_estimate)

@bp.route("/search", methods=["GET"])
def search():
    return flask.render_template("search.html")

@bp.route("/results", methods=["POST"])
def results():
    query_title = flask.request.form["title"]
    k = int(flask.request.form["k"])
    try:
        recommendations = flask.current_app.recommender.recommend(query_title, k)
        return flask.render_template("results.html", query=query_title, recommendations=recommendations)
    except HTTPError as e:
        flask.flash(f'Unable to parse title (HTTP code {e.response.status_code})', 'error')
        return flask.render_template("search.html")
    except FileNotFoundError:
        flask.flash('Building paper index, try again after a couple seconds', 'warning')
        return flask.render_template("search.html")
    
@bp.route("/publish", methods=["GET", "POST"])
def publish():
    if flask.request.method == 'GET':
        return flask.render_template("publish.html")
    else:
        # Prioritize arXiv ID, if it exists
        try:
            arxiv_id = flask.request.form['arxiv_id']
            if arxiv_id:
                flask.current_app.recommender.add_by_id(arxiv_id)
            else:
                REQUIRED_FIELDS = ['title', 'abstract', 'authors', 'url']
                for field in REQUIRED_FIELDS:
                    if not flask.request.form[field]:
                        flask.flash(f'Missing required field {field}', 'error')
                        flask.render_template('publish.html')
                flask.current_app.recommender.add_by_title(**flask.request.form)
            flask.flash('Paper successfully added')
        except (ValueError, TypeError) as e:
            flask.flash(e, 'error')
        except FileNotFoundError:
            flask.flash('Building paper index, try again after a couple seconds', 'warning')
    return flask.render_template('publish.html') 
                