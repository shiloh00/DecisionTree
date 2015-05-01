set xlabel "Size of Train Set"
set ylabel "Accuracy"
set title "Learning Curve on Unpruned Tree and Pruned Tree"
set xrange [0:51000]
set grid
set key box
set terminal png size 800,600
set output "new_plot.png"
plot "unpruned_curve.data" using 2:3 w lp pt 5 linewidth 2 title "Unpruned Tree", "pruned_curve.data" using 2:3 w lp pt 7 linewidth 2 title "Pruned Tree"
