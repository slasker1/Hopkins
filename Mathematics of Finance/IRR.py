import numpy as np

initialInvestment = -100  # Negative, since it results in an outflow of cash
cashFlows = [initialInvestment, 0, 0, 0, 300]
# Calculate the IRR
irr = round(np.irr(cashFlows), 4)
print("Internal rate of return:%3.4f" % irr)