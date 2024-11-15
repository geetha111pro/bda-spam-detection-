from flask import Flask, render_template, request
import email_spam_detection as e

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    # Initialize the default result for prediction
    sp = None
    if request.method == 'POST':
        email_content = request.form['email_content']
        # Get the spam prediction from the `predict_spam_or_ham` function
        sp = e.predict_spam_or_ham(email_content)
    
    # Render the template with the prediction result
    return render_template("index.html", spam_detect=sp)

if __name__ == "__main__":
    app.run(debug=True)
