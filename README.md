# tasks

1) generate answer with gk!
   - task1 '|GenerateAnswerGK|:'
2) generate answer withot gk
   - task2 '|GenerateAnswer|:'
3) generate gk by turn!
   - task3 '|GenerateGK|:'
4) retrieve gk by context!
   - task4q '|RetrieveGK-Q|:'
   - task4c '|RetrieveGK-C|:'
5) retrieve answer by context
   - task5q '|RetrieveAnswer-Q|:'
   - task5c '|RetrieveAnswer-C|:'

python3 main.py --config configs/t5.yml --config configs/tasks.yml
python3 test_main.py --config configs/t5.yml --config configs/tasks.yml