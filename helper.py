from xml.dom.expatbuilder import theDOMImplementation
import matplotlib.pyplot as plt
from IPython import display

plt.ion()
def plot(pol_scores, thi_score):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.plot(pol_scores, color = 'blue')
    plt.plot(thi_score, color = 'red')
    plt.ylim(ymin = 0)
    plt.text(len(pol_scores) - 1, pol_scores[-1], str(pol_scores[-1]))
    plt.text(len(thi_score) - 1, thi_score[-1], str(thi_score[-1]))
    plt.show(block = False)
    plt.pause(.1)
