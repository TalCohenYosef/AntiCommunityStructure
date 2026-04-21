1. קורא nodes.txt
2. מגריל one-hot ב-R2
3. שומר קובץ טקסט עם ה־features קורא edges.txt
4. בונה X, edge_index, edge_weight שומר הכל ל־JSON אחד מסודר

python level2/generate_input_to_gnn.py


הקוד הזה הוא שלב הכנת הקלט ל־GNN בפרויקט שלך: הוא לוקח את הגרף ששמרת כ־nodes.txt ו־edges.txt, מגריל לכל קודקוד פיצ'ר התחלתי one-hot ב-R
2
, ואז ממיר את הכל לפורמט שהמודל יוכל לטעון בהמשך.

מה כל חלק עושה:

load_nodes_with_random_one_hot(...)
קורא את nodes.txt, שומר את רשימת הקודקודים, בונה מיפוי מ־node_id לאינדקס מספרי, ומגריל לכל קודקוד וקטור התחלתי [1,0] או [0,1]. זה בעצם יוצר את ה־features ההתחלתיים של הקודקודים.

save_nodes_with_init(...)
שומר לקובץ טקסט את הקודקודים יחד עם שני הרכיבים של הווקטור שהוגרל להם. זה טוב לבדיקה ידנית, כדי לראות מה כל קודקוד קיבל בהתחלה.

load_edges(...)
קורא את edges.txt, ממיר כל קשת משמות של קודקודים לאינדקסים מספריים, ובונה:

edge_index — רשימת קשתות בפורמט שמתאים ל־GNN
edge_weight — משקל לכל קשת
בנוסף, בגלל שהגרף לא מכוון, הוא שומר כל קשת בשני הכיוונים.

save_gnn_input_json(...)
שומר את כל הקלט המאורגן ל־JSON אחד:

node_ids
x_init
edge_index
edge_weight
זה כבר כמעט הקלט הישיר שהקוד הבא יטען ל־PyTorch Geometric.

החלק של if __name__ == "__main__":
זה ה־pipeline עצמו:

קורא את level1/nodes.txt ו־level1/edges.txt
מגריל one-hot לכל קודקוד
שומר קובץ בדיקה nodes_with_init.txt
בונה edge_index ו־edge_weight
שומר הכל ל־gnn_input.json
כלומר זה השלב שמכין מהגרף הגולמי שלך קלט מסודר למודל.