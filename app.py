from flask import Flask, request, jsonify, render_template
import os
import analytics
# import tensorflow as tf



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/sub',methods=['POST'])
def sub():
    if request.method == 'POST':
        name = request.form["crypto"]
        type = request.form["types"]
        print(name)
        analytics.plot_bitcoin_data(name,type)
    return render_template('sub.html',n=name,t =type)


if __name__ == '__main__':
    app.debug=True
    app.run()
