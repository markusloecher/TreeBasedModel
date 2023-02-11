## Open Questions so far

#### Allgemeine Fragen:
 
-	Du hattest nach unseren meetings meistens/immer eine sehr gute summary geschickt, die auch next action bullets enthielt. Diese gibt es nur als emails, nicht als Dokument irgendwo af drive oder github, richtig ?
-	Die Notation in Gleichungen (12), (13) ist etwas verwirrend. Die „hats“ sind ja eigentlich für estimates gedacht, aber hier werden doch lediglich die y-Werte eingesetzt, oder? (Und ein Mittelwert mit dem „Strich“ drüber enthält ja keinen Index mehr) Und wollten wir nicht den oob-MSE berechnen, der sich auf dieselbe node prediction wie inbag bezieht? Also eher so etwas:
$$MSE_{in} = \frac{1}{n_{in}-1} \sum_{i=1}^{n_{in}}{(y_{in,i} - \overline{y_{in}})}$$
$$MSE_{oob} = \frac{1}{n_{oob}-1} \sum_{i=1}^{n_{oob}}{(y_{oob,i} - \overline{y_{in}})}$$
- Schön zu sehen, dass Du die *Finite sample correction in Gini impurity* ("If k=1, impurity is weighted by n/(n-1)") implementiert hast. Wie schwierig wäre es wohl, den code so zu modifizieren, dass das $k$ im correction factor $n/(n-k)$ von node zu node verschieden ist (und zwar gleich der jeweiligen tree depth)?
- No introduction to the two other Friedman data sets 2 and 3 ?
- Die doch sehr verschiedenen lambdas in Figure 

              
 
#### Code related:

-	Ist das folgende Verständnis korrekt:
Das extra shrinkage wird m.E. in Zeilen 698-699 (DecisionTree.py) angewandt?:
```python
cum_sum += ((current_node.value - parent_node.value) / (1 + HS_lambda/parent_node.samples)) * m_nodes[node_id]
In Zeilen 479-487 (RandomForest.py)
```
werden diese m_nodes berechnet. Und zwar ganz konkret in Zeilen 153-154 (SmoothShap.py):
```python
mse_inbag = mean_squared_error(node_val_pop1, pop_1)
mse_oob = mean_squared_error(node_val_pop2, pop_2)
```
-	Wann haben wir noch mal „test m_shrinkage of lambda instead of expected term“ In Zeilen 702-704(DecisionTree.py) untersucht?

- Für die benchmarks: könnten wir Zeit sparen, den "vanilla random forest" mit der sklearn Version zu ersetzen ?

- In `Run_PredPerf_experiment`: 
    * ist die Hauptmotivation für die 10 test splits, dass wir einen "sem" (standard error of the mean) ausrechnen können? Aber es gibt nur einen train/test split, nicht wahr.
    * beeindruckend, dass hier anscheinend die Datensätze in parallel evaluiert werden ? (mit "Pool")
    * werden die simulated data separat von den "data on disk" evaluiert? 
    * wenn ich also nur einen Datensatz noch einmal evaluieren möchte


#### Module Structure

- Wäre es nicht klug, **alle** benutzten Funktionen (also auch Hilfsfunktionen wie `simulate_data_strobl`, etc.) im Modul zu verankern ?

#### I would like to "quickly":

- find the best lambda and prediction for one particular data set, e.g. *heart*. Is there a function for that or should I/do I need to modify the notebook *Experiment_PredPerf.ipynb* ?
