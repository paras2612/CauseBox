import sqlite3
# Create a SQL connection to our SQLite database
con = sqlite3.connect('C:\\Users\\coole\\Downloads\\perfect_match-master\\perfect_match-master\\perfect_match\\apps\\jobs.db',check_same_thread=False,detect_types=sqlite3.PARSE_DECLTYPES)

query = "SELECT " \
                "{columns} " \
                "FROM {table_name} " \
                "WHERE rowid = ?;".format(table_name="jobs",
                                          columns="rowid,x")
for row in con.execute(query, (1,)).fetchone():
    print(row)