extern "C" {
#include <evaluate.h>
}

#include <spatialindex/SpatialIndex.h>

using SpatialIndex::INode;
using SpatialIndex::IData;
using SpatialIndex::IVisitor;
using SpatialIndex::ISpatialIndex;
using SpatialIndex::Point;


struct SearchResult {
	int cell;
	SearchResult(int c) : cell(c) {}
};

class Visitor : public IVisitor {
public:
	Visitor(struct Function *f, double *x, inside_predicate p, void *data)
		: m_f(f), m_x(x), m_p(p), m_data(data) {}

	virtual void visitNode(const INode&);
	virtual void visitData(const IData&);
	virtual void visitData(std::vector<const IData*>&);

private:
	struct Function *m_f;
	double *m_x;
	inside_predicate m_p;
	void *m_data;
};

void Visitor::visitNode(const INode& in)
{
	(void) in;
}

void Visitor::visitData(const IData& in)
{
	int cell = in.getIdentifier();
	if ((*m_p)(m_data, m_f, cell, m_x)) {
		// Do not use exceptions for flow control
		throw SearchResult(cell);
	}
}

void Visitor::visitData(std::vector<const IData*>& v)
{
	for (size_t i = 0; i < v.size(); i++) {
		visitData(*v[i]);
	}
}

extern "C" int locate_cell(struct Function *f,
			   double *x,
			   int dim,
			   inside_predicate try_candidate,
			   void *data_)
{
	if (f->sidx) {
		SpatialIndex::ISpatialIndex *spatial_index =
			reinterpret_cast<SpatialIndex::ISpatialIndex *>(f->sidx);

		Visitor visitor(f, x, try_candidate, data_);
		Point point(x, dim);

		// Do not use exceptions for flow control
		try {
			spatial_index->pointLocationQuery(point, visitor);
		} catch (const SearchResult& ex) {
			return ex.cell;
		}
	} else {
		for (int c = 0; c < f->n_cols * f->n_layers; c++)
			if ((*try_candidate)(data_, f, c, x))
				return c;
	}
	return -1;
}
