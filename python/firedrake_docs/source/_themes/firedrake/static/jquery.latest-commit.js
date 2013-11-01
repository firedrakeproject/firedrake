// Based on http://github.com/skulbuny/jquery.latest-commit.js by Sean Clayton
jQuery(document).ready(function($){
  $('.latest-commit').each(function(){ // Attach to any div/section/whatever with this class
    var $container = $(this), $commit,
      repo = $container.data('github'),
      username = repo.split('/')[0],
      repoName = repo.split('/')[1],
      userUrl = "http://github.com/" + username, // Gets your user url
      repoUrl = "http://github.com/" + username + '/' + repoName; // Gets your repo url
    $.ajax({
      url: 'https://api.github.com/repos/' + repo + '/commits?per_page=' + $container.data('commits'),
      dataType: 'jsonp',
      success: function(results) {
        for (i = 0; i < results.data.length; ++i) {
          $commit = $(
            '<div class="commit">' + // Needs to be wrapped.
            // ADD DESIRED CLASSES TO HTML TAGS BELOW!
            '<img class="commit-author-img" src="#" />' + // Commit author image
            '<div class="commit-meta">' +
            '<div><a class="commit-link" href="#"></a></div>' + // First line of commit message
            '<a class="commit-author" href="#" target="_blank"></a>' + // Link to commit author
            ' authored at <span class="commit-date"></span>' + // Outputs the commit date
            '</div>' +
            '</div>'
          );
          var repo = results.data[i];
          var commitUrl = repo.html_url; // Grabs URL of the commit
          $commit.find('.commit-author-img').attr('src', repo.author.avatar_url); // Add commit author avatar image
          $commit.find('.commit-link').attr('href',commitUrl).text(repo.commit.message.split("\n")[0]); // Adds link to commit and commit SHA
          $commit.find('.commit-author').attr('href', repo.author.html_url).text(repo.commit.author.name); // Outputs commit author name
          $commit.find('.commit-date').text(new Date(repo.commit.author.date).toLocaleString()); // Outputs commit date
          $commit.appendTo($container);
        }
        $('.commit:even').addClass('even');
      }
    });
  });
});
