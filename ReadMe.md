Tema 2: Mouse vs Cat

Reprezentarea: pentru reprentarea am creat clasa Map care retine
date citite din fisierul de congfigurarea si are si fuctii ajutatorea
pentru algoritmul q-learning si sarsa, ea se gaseste in fiserul Map. <br> Pentru afisarea jocului am
folosit pygame. </br>
(clasa Display din src/display)Cum pentru exploare era la alegrea noastra, am implementat trei
startegi: se alege cele
care au fost vizitate cel mai putin, se foloseste o varinta modificat de uct, de la MCTS, alegandu-se
maximul sau se
alege probabilistic, calculand probabilitatile folosind softmax. </br>
Pentru exploare/exploate am doua strategi: aleg cu o probabilitate de epsilor intre
random/max_first; aleg actionea
probabilistic, probalitatile sunt calculate cu ajutorul functie softmax. </br>
