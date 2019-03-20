import jinja2
import matplotlib as plt

leaderboard_template="""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
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
  background-color: #f1f1f1;
}
</style>
</head>
<body>


{% for participant in participants %}
  <button class="collapsible">{{loop.index}} : {{participant["best_map"]}} {{participant["name"]}}</button>
    <div class="content">
        <h5>Method description:</h5>
        <p>{{participant["description"]}}</p>
        <hr>
        {% for submission in participant["submissions"] %}
            <button class="collapsible">{{loop.index}} : Submitted on {{submission["date"]}} {{submission["map"]}} %</button>
            <div class="content">
            <h5>Submission Performance:</h5>
            <table border=1>
                <tr><th>mAP %</th> <th> Precision %</th> <th> Recall %</th> <th> F-Score %</th> </tr>
                <tr><th>{{submission["map"]}}</th> <th>{{submission["pr"]}}</th> <th>{{submission["rec"]}}</th> <th>{{submission["fm"]}}</th> </tr>
            </table>
            <br>
            <img src="{{submission['roc_svg']}}"/>
            <hr>
            </div>
        {% endfor %}
   </div>

{% endfor %}

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
</body>
</html>
"""


if __name__=="__main__":
    template=jinja2.Template(leaderboard_template)
    participants=[
        {
            "name":"Team John Do!",
            "best_map":"34.7%",
            "description": "N/A",
            "submissions":[
                {"date":"NA",
                 "map":"34.7",
                 "pr": "14.7",
                 "rec": "14.7",
                 "fm": "13.11",
                 "roc_svg": "/tmp/img.svg",
                 },
                {"date": "NA",
                 "map": "24.7",
                 "pr": "14.7",
                 "rec": "14.7",
                 "fm": "15.11",
                 "roc_svg": "/tmp/img.svg",
                 }
            ]
        },
    {
        "name": "Team Jane Do!",
        "best_map": "34.7",
        "description":"A long description people give optionally. "* 20,
        "submissions":[
            {"date": "NA",
             "map": "34.7",
             "pr": "14.7",
             "rec": "14.7",
             "fm":"19.11",
             "roc_svg": "/tmp/img.svg",
             },
            {"date": "NA",
             "map": "24.7",
             "pr": "14.7",
             "rec": "14.7",
             "fm": "29.11",
             "roc_svg": "/tmp/img.svg",
             }
        ]
    }
    ]

    open("/tmp/index.html","w").write(template.render(participants=participants))