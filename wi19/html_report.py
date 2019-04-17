import jinja2

leaderboard_template="""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.all{
max-width: 1000px;
background-color: #e1e1e1;
}
.collapsible {
  background-color: #777;
  color: white;
  cursor: pointer;
  padding: 18px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
}

.active, .collapsible:hover {
  background-color: #555;
}

.content {
  padding: 0 18px;
  display: none;
  overflow: hidden;
  background-color: #d1d1d1;
  max-width: 982px;
}

th, td {
  padding: 8px;
  text-align: left;
}

tr:hover {background-color: #e5e5e5;}

</style>
</head>
<body>
<div class="all">
<div align="center" class="head">
<h3>ICDAR2019 COMPETITION ON IMAGE RETRIEVAL FOR HISTORICAL HANDWRITTEN DOCUMENTS
<br>
<a href="https://clamm.irht.cnrs.fr/icdar2019-hdrc-ir/">ICDAR-2019-HDRC-IR</a>
</h3>
</div>
<div align="center">
    <img src="{{all_participants['participants_svg']}}"/>

    <table>
    <tr><th> # </th> <th>Team Name</th> <th>Best mAP up to now</th> <th>Last submission mAP</th> </th>
    
    {% for n in range(all_participants['names']| length) %}
    <tr> <td>{{n}}</td> <td> {{ all_participants['names'][n]}} </td> <td>{{100*all_participants['best_maps'][n]|round(4, 'floor')}}%</td> <td>{{100*all_participants['last_maps'][n]|round(4, 'floor') }} %</td> </tr> 
    {% endfor %}
    </table>
</div>
{% for participant in all_participants['participants'] %}
<button class="collapsible">{{participant['name']}}</button>
<div class="content">
    <h4>Description:</h4>
    {{ participant['description'] }}
    <hr>
    <h4>Submissions:</h4>
    {% for submission in participant['submissions'] %}
    <button class="collapsible">{{submission['date']}}   mAP: <b>{{100*submission['map']|round(4, 'floor')}} %</b></button>
    <div class="content" align="center">
        <img src="{{submission['roc_svg']}}"/>
        <br>
        </table>
        <tr> <td>mAP:</td> <td>{{100*submission['map']|round(4, 'floor')}}</td> </tr>
        <tr> <td>Precision:</td> <td>{{100*submission['pr']|round(4, 'floor')}}</td> </tr>
        <tr> <td>Recall:</td> <td>{{100*submission['rec']|round(4, 'floor')}}</td> </tr>
        <tr> <td>F-score:</td> <td>{{100*submission['fm']|round(4, 'floor')}}</td> </tr>
        </table>
    </div>
    {% endfor %}
</div>
{% endfor %}

<div class="footer">
Report compiled in {{all_participants['duration']|round(5, 'floor')}} sec.
<br>
Report compiled on {{all_participants['date'] }}.
<br>
Utility writen by anguelos (dot) nicolaou (at) gmail (dot) com <a href="https://github.com/anguelos/wi19_evaluate">source code</a>
</div>


<script>
var coll = document.getElementsByClassName("collapsible");
var i;
for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
</script>
</div>
</body>
</html>
"""

def render_leaderboard(root,all_participants):
    template = jinja2.Template(leaderboard_template)
    open("{}/index.html".format(root), "w").write(template.render(all_participants=all_participants))

def render_submission(root,submission):
    raise NotImplementedError()
