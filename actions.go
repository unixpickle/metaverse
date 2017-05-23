package metaverse

import (
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// FlashKeyMasks stores, for each Flash environment, the
// keys which perform meaningful actions.
var FlashKeyMasks = map[string]StringSet{
	"flashgames.DuskDrive-v0": NewStringSet("ArrowUp", "ArrowLeft",
		"ArrowRight", "space"),
	"flashgames.EasterEggsChallenge-v0": NewStringSet(),
}

// FlashPointerInfo stores, for each Flash environment,
// the allowed pointer events.
// For games with no pointer events, the entry is nil.
var FlashPointerInfo = map[string]*PointerInfo{
	"flashgames.DuskDrive-v0": nil,
	"flashgames.EasterEggsChallenge-v0": &PointerInfo{
		PointerRect: &Rect{X: 18, Y: 84, Width: 760, Height: 570},
		NoClick: []*Rect{
			{X: 17, Y: 128, Width: 48, Height: 47},
			{X: 212, Y: 128, Width: 59, Height: 106},
		},
	},
}

// AllFlashKeys returns all the keys that are useful in at
// least one Flash game.
func AllFlashKeys() StringSet {
	var sets []StringSet
	for _, s := range FlashKeyMasks {
		sets = append(sets, s)
	}
	return StringSetUnion(sets...)
}

// ActionSpace specifies a set of events to use on a
// particular Universe environment.
type ActionSpace struct {
	// Keys contains all the keys the agent knows about.
	Keys StringSet

	// KeyMask contains all the keys that the agent is
	// able to press.
	// If nil, all elements of Keys are allowed.
	//
	// You might want to set this to a subset of Keys if
	// you want to use an agent on multiple games, where
	// each game has a different set of allowed keys.
	KeyMask StringSet

	// Pointer specifies whether or not to let the agent
	// attempt to send pointer events.
	//
	// You might want to set this even if PointerInfo is
	// nil, for example to use an agent that must work on
	// games with and without pointer events.
	Pointer bool

	// PointerMask defines where to allow pointer events.
	// If nil, pointer events are dropped everywhere.
	// If Pointer is false, then PointerInfo has no
	// meaningful effect.
	PointerInfo *PointerInfo
}

// VecToObj creates a Universe-compatible action object
// from a sampled action vector.
//
// The returned object will not contain any disallowed
// actions, such as those blocked by a.NoClick.
func (a *ActionSpace) VecToObj(vec anyvec.Vector) interface{} {
	c := vec.Creator()
	ops := c.NumOps()

	var res []interface{}

	for i, keyName := range a.Keys.Slice() {
		if a.KeyMask != nil && !a.KeyMask.Contains(keyName) {
			continue
		}
		num := anyvec.Sum(vec.Slice(i, i+1))
		pressed := !ops.Equal(num, c.MakeNumeric(0))
		res = append(res, []interface{}{"KeyEvent", keyName, pressed})
	}

	if a.Pointer && a.PointerInfo != nil {
		area := a.PointerInfo.PointerRect
		halfWidth, halfHeight := float64(area.Width)/2, float64(area.Height)/2
		x := numToFloat64(anyvec.Sum(vec.Slice(vec.Len()-3, vec.Len()-2)))
		y := numToFloat64(anyvec.Sum(vec.Slice(vec.Len()-2, vec.Len()-1)))
		mask := numToFloat64(anyvec.Sum(vec.Slice(vec.Len()-1, vec.Len())))

		// X and Y coordinates are bounded in [-1, 1].
		rawX := int(essentials.Round((x*halfWidth)+halfWidth)) + area.X
		rawY := int((y*halfHeight)+halfHeight) + area.Y
		rawX, rawY = area.Clip(rawX, rawY)
		for _, noClick := range a.PointerInfo.NoClick {
			if noClick.Contains(rawX, rawY) {
				mask = 0
			}
		}

		res = append(res, []interface{}{"PointerEvent", rawX, rawY, mask})
	}

	return res
}

// ParamSize returns the size of parameter vectors for the
// action space.
func (a *ActionSpace) ParamSize() int {
	if a.Pointer {
		return len(a.Keys) + 5
	}
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

func (a *ActionSpace) actionSpace() *anyrl.Tuple {
	res := &anyrl.Tuple{
		Spaces:      []interface{}{&anyrl.Bernoulli{}},
		ParamSizes:  []int{len(a.Keys)},
		SampleSizes: []int{len(a.Keys)},
	}
	if a.Pointer {
		// X, Y, and clicked.
		res.Spaces = append(res.Spaces, &anyrl.Gaussian{}, &anyrl.Bernoulli{})
		res.ParamSizes = append(res.ParamSizes, 4, 1)
		res.SampleSizes = append(res.SampleSizes, 2, 1)
	}
	return res
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

// StringSetUnion computes the union of string sets.
func StringSetUnion(sets ...StringSet) StringSet {
	res := StringSet{}
	for _, s := range sets {
		for k, v := range s {
			res[k] = v
		}
	}
	return res
}

// Contains checks if a string is in the set.
func (s StringSet) Contains(elem string) bool {
	_, ok := s[elem]
	return ok
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

// PointerInfo contains information about where pointer
// events may occur.
type PointerInfo struct {
	// PointerRect is the region to which pointer events
	// should be constrained.
	PointerRect *Rect

	// NoClick contains regions where clicking should be
	// forbidden.
	NoClick []*Rect
}

// Rect is a rectangular screen region.
type Rect struct {
	X      int
	Y      int
	Width  int
	Height int
}

// Contains checks if a point is inside the rectangle.
func (r *Rect) Contains(x, y int) bool {
	return x >= r.X && x < r.X+r.Width && y >= r.Y && y < r.Y+r.Height
}

// Clip clips the point to be inside of the rectangle.
func (r *Rect) Clip(x, y int) (newX, newY int) {
	newX = essentials.MaxInt(essentials.MinInt(x, r.X+r.Width-1), r.X)
	newY = essentials.MaxInt(essentials.MinInt(y, r.Y+r.Height-1), r.Y)
	return
}

func numToFloat64(num anyvec.Numeric) float64 {
	switch num := num.(type) {
	case float32:
		return float64(num)
	case float64:
		return num
	default:
		panic("unsupported numeric type")
	}
}
