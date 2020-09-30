# Epstein Civil Violence Model

## Summary

This model is based on Joshua Epstein's simulation of how civil unrest grows and is suppressed. Citizen agents wander the grid randomly, and are endowed with individual risk aversion and hardship levels; there is also a universal regime legitimacy value. There are also Cop agents, who work on behalf of the regime. Cops arrest Citizens who are actively rebelling; Citizens decide whether to rebel based on their hardship and the regime legitimacy, and their perceived probability of arrest. 

The model generates mass uprising as self-reinforcing processes: if enough agents are rebelling, the probability of any individual agent being arrested is reduced, making more agents more likely to join the uprising. However, the more rebelling Citizens the Cops arrest, the less likely additional agents become to join.

## How to Run

To run the model in jupyter notebook, open ``EpsteinCivilViolence.ipynb`` from this directory. e.g.

```
    EpsteinCivilViolence.ipynb
``` 
To run the model in python import the necessary functions from the civil violence directory, then add model parameters and run the model. e.g.

```
    $ python from epstein_civil_violence.agent import Citizen, Cop
    $ python from epstein_civil_violence.model import EpsteinCivilViolence
    $ python model = EpsteinCivilViolence(height=40, 
                           width=40, 
                           citizen_density=.7, 
                           cop_density=.074, 
                           citizen_vision=7, 
                           cop_vision=7, 
                           legitimacy=.8, 
                           max_jail_term=1000, 
                           initial_unemployment_rate = 0.2,
                           corruption_level = 0.1,
                           susceptible_level = 0.6,
                           max_iters=200) # cap the number of steps the model takes
    $ python model.run_model()
    
``` 
