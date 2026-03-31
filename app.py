import streamlit as st
import pyagrum as gum
import pyagrum.lib.notebook as gnb
import matplotlib.pyplot as plt

st.title("Interactive Bayesian Network")

# 1. Setup Model
bn = gum.fastBN("Cloudy->Sprinkler;Cloudy->Rain;Sprinkler->WetGrass;Rain->WetGrass")
bn.cpt("Cloudy").fillWith([0.5, 0.5])
bn.cpt("Rain")[0] = [0.8, 0.2]
bn.cpt("Rain")[1] = [0.2, 0.8]

# 2. Sidebar Sliders for User Input
st.sidebar.header("Adjust Evidence")
prob_cloudy = st.sidebar.slider("Probability of Cloudy", 0.0, 1.0, 0.5)

# Update model with slider value
bn.cpt("Cloudy").fillWith([1 - prob_cloudy, prob_cloudy])

# 3. Inference & Visualization
st.subheader("Network Structure")
fig, ax = plt.subplots()
# We use the standard matplotlib export for Streamlit
gnb.showBN(bn, matplotlib=True) 
st.pyplot(plt.gcf())

st.subheader("Inference Results")
ie = gum.LazyPropagation(bn)
ie.makeInference()
st.write(f"Current Probability of Rain: {ie.posterior('Rain')[1]:.2f}")
