Python Bank Portal with integrated BankBot
------------------------------------------
How to run locally:
1. Extract the ZIP and cd into the project folder.
2. (Optional) create a virtualenv and activate it.
3. Install requirements:
   pip install -r requirements.txt
4. Run:
   python app.py
5. Open http://127.0.0.1:5000 in your browser.

Notes:
- Your uploaded BankBot content was extracted into the 'bankbot' folder. The dashboard embeds the bot at /bankbot/ via an iframe.
- Ensure your BankBot frontend has an index.html (or rename its main html to index.html) so the iframe can load it.
- Database: a SQLite file 'bank.db' is created automatically with a sample user (email: test@example.com, password: testpass, account: ACC1001).