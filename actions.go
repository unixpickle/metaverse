package metaverse

import (
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
)

// FlashActionSpaces stores the relevant actions for each
// Flash game environment.
var FlashActionSpaces = map[string]*ActionSpace{
	"flashgames.DuskDrive-v0": &ActionSpace{
		Keys: NewStringSet("ArrowUp", "ArrowLeft", "ArrowRight", "space"),
	},
}

// ActionSpace specifies a set of events to use on a
// particular Universe environment.
type ActionSpace struct {
	Keys StringSet
}

// ActionSpaceUnion computes the union action spaces.
// The union contains all of the actions from all of the
// action spaces.
func ActionSpaceUnion(spaces ...*ActionSpace) *ActionSpace {
	keys := StringSet{}
	for _, s := range spaces {
		for key := range s.Keys {
			keys[key] = struct{}{}
		}
	}
	return &ActionSpace{Keys: keys}
}

// FlashActionSpace returns the union of the ActionSpaces
// for all supported Flash games.
func FlashEvents() *ActionSpace {
	var list []*ActionSpace
	for _, s := range FlashActionSpaces {
		list = append(list, s)
	}
	return ActionSpaceUnion(list...)
}

// ParamSize returns the size of parameter vectors for the
// action space.
func (a *ActionSpace) ParamSize() int {
	return len(a.Keys)
}

// Sample samples from the action space given a batch of
// parameter vectors.
func (a *ActionSpace) Sample(params anyvec.Vector, batch int) anyvec.Vector {
	return a.actionSpace().Sample(params, batch)
}

// LogProb computes the log probabilities of samples given
// the parameters that produced them.
func (a *ActionSpace) LogProb(params anydiff.Res, out anyvec.Vector,
	batchSize int) anydiff.Res {
	return a.actionSpace().LogProb(params, out, batchSize)
}

// KL computes the KL divergences between two batches of
// action distributions.
func (a *ActionSpace) KL(params1, params2 anydiff.Res, batchSize int) anydiff.Res {
	return a.actionSpace().KL(params1, params2, batchSize)
}

// Entropy computes the entropy of the distributions.
func (a *ActionSpace) Entropy(params anydiff.Res, batchSize int) anydiff.Res {
	return a.actionSpace().Entropy(params, batchSize)
}

func (a *ActionSpace) actionSpace() *anyrl.Bernoulli {
	return &anyrl.Bernoulli{}
}

// StringSet is a set of strings.
// A string is in the set if it exists as a key.
type StringSet map[string]struct{}

// NewStringSet creates a set with the given strings.
func NewStringSet(s ...string) StringSet {
	res := StringSet{}
	for _, str := range s {
		res[str] = struct{}{}
	}
	return res
}

// Slice turns the set into a sorted list.
func (s StringSet) Slice() []string {
	var res []string
	for elem := range s {
		res = append(res, elem)
	}
	sort.Strings(res)
	return res
}
