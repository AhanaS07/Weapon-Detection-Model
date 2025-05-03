
To run outside of docker
first run "pip install -r requirements.txt" (pnly for first times)
then Double click on app.py and click on play button on top-right.
Then go to http://localhost:5000 on the browser

To run within docker
Open docker desktop in admin user
Right click in project explorer in vscode and click on "open in integrated terminal"
Execute below two commands one after the other
    1.docker build -t tensorflow-image .
    2.docker run -p 5000:5000 tensorflow-image
Then go to http://localhost:5000 on the browser
