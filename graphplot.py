import graphviz as gr
import matplotlib.pyplot as plt


g = gr.Digraph()
g.node(name="seasonality",color= "gainsboro",  label="seasonality", style="filled")
g.node(name="trend", color= "gainsboro", label="trend", style='filled')
g.node(name="media", label="media", color="forestgreen", style="filled")
g.node(name="jackpot", label="jackpot", color="firebrick3", style="filled")
g.node(name="sales", color= "gold",label="sales", style="filled")
g.edge(tail_name="jackpot", head_name="media")
g.edge(tail_name="jackpot", head_name="sales")
g.edge(tail_name="media", head_name="sales")
g.edge(tail_name="seasonality", head_name="sales")
g.edge(tail_name="trend", head_name="sales")
g.format = "png"
g.render("models/causal_graph")