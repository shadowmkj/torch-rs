use crate::tensor::Tensor;

pub struct Iter<'a, const D: usize> {
    tensor: &'a Tensor<D>,
    coords: [usize; D],
    remaining: usize,
}

impl<'a, const D: usize> Iterator for Iter<'a, D> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let val = self.tensor.get(self.coords).ok()?;
        self.remaining -= 1;

        if D > 0 {
            for dim in (0..D).rev() {
                self.coords[dim] += 1;
                if self.coords[dim] < self.tensor.shape[dim] {
                    break;
                }
                self.coords[dim] = 0;
            }
        }

        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const D: usize> Tensor<D> {
    pub fn iter(&self) -> Iter<'_, D> {
        Iter {
            tensor: self,
            coords: [0; D],
            remaining: self.numel(),
        }
    }
}

impl<'a, const D: usize> IntoIterator for &'a Tensor<D> {
    type Item = f32;
    type IntoIter = Iter<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
