var pagination = require('hexo-pagination');

hexo.extend.generator.register('sticky', function(locals){
  var config = this.config;
  var posts = locals.posts.sort(config.index_generator.order_by);
  
  posts.data = posts.data.sort(function (a, b) {
	var sticky_a = a.sticky || 0;
	var sticky_b = b.sticky || 0;
	return sticky_b-sticky_a
  });
  
  var paginationDir = config.pagination_dir || 'page';
  var path = config.index_generator.path || '';

  return pagination(path, posts, {
    perPage: config.index_generator.per_page,
    layout: ['index', 'archive'],
    format: paginationDir + '/%d/',
    data: {
      __index: true
    }
  });
});