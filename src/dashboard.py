from flask import Flask, render_template
from database import initialize_database, get_recent_investigations
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

initialize_database()


@app.route("/")
def dashboard():
    investigations = get_recent_investigations(50)

    return render_template(
        "dashboard.html",
        investigations=investigations
    )


@app.route("/investigation/<int:investigation_id>")
def investigation_details(investigation_id):
    investigations = get_recent_investigations(100)

    investigation = next(
        (
            item
            for item in investigations
            if item["id"] == investigation_id
        ),
        None
    )

    if investigation is None:
        return "Investigation not found", 404

    return render_template(
        "investigation.html",
        investigation=investigation
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )