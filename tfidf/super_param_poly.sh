for i in {2..20..1}; do
  echo $i
  python3 tfidf.py True $i > ./out/out.poly_$i.txt
done