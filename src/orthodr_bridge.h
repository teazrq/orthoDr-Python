// Python bridge for orthoDr. GPL-3.0-or-later; upstream core GPL-2.0-or-later.
// These are data-conversion adapters only; no R headers or R runtime are used.
#pragma once
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace orthodr_py {

template<class T> py::object pack(const T& value) {
  if constexpr (arma::is_arma_type<T>::value) {
    using E = typename T::elem_type;
    const arma::Mat<E> evaluated(value);
    if constexpr (T::is_col || T::is_row) {
      py::array_t<E> out(evaluated.n_elem);
      std::copy(evaluated.begin(), evaluated.end(), out.mutable_data());
      return out;
    } else {
      py::array_t<E> out({static_cast<py::ssize_t>(evaluated.n_rows),
                          static_cast<py::ssize_t>(evaluated.n_cols)});
      auto data = out.template mutable_unchecked<2>();
      for (arma::uword r = 0; r < evaluated.n_rows; ++r)
        for (arma::uword c = 0; c < evaluated.n_cols; ++c)
          data(r, c) = evaluated(r, c);
      return out;
    }
  } else {
    return py::cast(value);
  }
}

struct NamedValue { std::string name; py::object value; };
struct Named {
  std::string name;
  explicit Named(const char* key) : name(key) {}
  template<class T> NamedValue operator=(const T& value) const {
    return {name, pack(value)};
  }
};

// A named upstream List is a dict; an append-only trace is a Python list.
class List {
  py::object value_;
 public:
  List() : value_(py::dict()) {}
  explicit List(py::object value) : value_(std::move(value)) {}
  py::object object() const { return value_; }
  std::size_t size() const { return py::len(value_); }
  template<class... T> static List create(const T&... values) {
    py::dict out;
    ((out[py::str(values.name)] = values.value), ...);
    return List(out);
  }
  class Item {
    List& owner_; py::object key_;
   public:
    Item(List& owner, py::object key) : owner_(owner), key_(std::move(key)) {}
    template<class T> Item& operator=(const T& value) {
      owner_.value_[key_] = pack(value); return *this;
    }
    operator List() const {
      return List(py::reinterpret_borrow<py::object>(owner_.value_[key_]));
    }
  };
  Item operator[](const char* key) { return Item(*this, py::str(key)); }
  Item operator[](std::size_t index) { return Item(*this, py::int_(index)); }
  Item operator[](int index) { return Item(*this, py::int_(index)); }
  void push_back(const List& value) {
    if (py::isinstance<py::dict>(value_) && size() == 0) value_ = py::list();
    value_.cast<py::list>().append(value.object());
  }
};

class NumericVector {
  arma::vec value_;
 public:
  NumericVector() = default;
  explicit NumericVector(arma::vec value) : value_(std::move(value)) {}
  explicit NumericVector(py::handle value) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) throw py::type_error("Expected a numeric vector");
    if (array.ndim() > 2 || (array.ndim() == 2 && array.shape(0) != 1 && array.shape(1) != 1))
      throw py::value_error("Expected a one-dimensional numeric vector");
    value_ = arma::vec(array.data(), array.size());
  }
  std::size_t size() const { return value_.n_elem; }
  double operator[](std::size_t i) const { return value_[i]; }
  const arma::vec& vector() const { return value_; }
  static bool is_na(double value) { return std::isnan(value); }
  static NumericVector create(double lower, double upper) {
    return NumericVector(arma::vec({lower, upper}));
  }
};

template<class T> class Nullable {
  std::optional<T> value_;
 public:
  Nullable(std::nullptr_t = nullptr) {}
  Nullable(const T& value) : value_(value) {}
  explicit Nullable(py::handle value) {
    if (!value.is_none()) value_.emplace(value);
  }
  bool isNotNull() const { return value_.has_value(); }
  bool isNull() const { return !value_.has_value(); }
  const T& get() const { return value_.value(); }
};

template<class T> T as(const NumericVector& value) { return T(value.vector()); }
inline void stop(const std::string& message) { throw std::invalid_argument(message); }
inline void warning(const std::string& message) {
  if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) != 0)
    throw py::error_already_set();
}
inline void checkUserInterrupt() {
  if (PyErr_CheckSignals() != 0) throw py::error_already_set();
}
inline constexpr double missing_value = std::numeric_limits<double>::quiet_NaN();

} // namespace orthodr_py

namespace pybind11::detail {
template<class T, class Matrix> struct arma_caster {
  PYBIND11_TYPE_CASTER(Matrix, const_name("numpy.ndarray"));
  bool load(handle src, bool) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
    if (!array) return false;
    if constexpr (Matrix::is_col || Matrix::is_row) {
      if (array.ndim() != 1 && !(array.ndim() == 2 && (array.shape(0) == 1 || array.shape(1) == 1))) return false;
      value = Matrix(array.data(), array.size());
    } else {
      if (array.ndim() != 2) return false;
      value.set_size(array.shape(0), array.shape(1));
      const auto data = array.template unchecked<2>();
      for (arma::uword r = 0; r < value.n_rows; ++r)
        for (arma::uword c = 0; c < value.n_cols; ++c) value(r, c) = data(r, c);
    }
    return true;
  }
  static handle cast(const Matrix& value, return_value_policy, handle) {
    return orthodr_py::pack(value).release();
  }
};
template<class T> struct type_caster<arma::Mat<T>> : arma_caster<T, arma::Mat<T>> {};
template<class T> struct type_caster<arma::Col<T>> : arma_caster<T, arma::Col<T>> {};
template<class T> struct type_caster<arma::Row<T>> : arma_caster<T, arma::Row<T>> {};
template<> struct type_caster<orthodr_py::List> {
  PYBIND11_TYPE_CASTER(orthodr_py::List, const_name("dict"));
  bool load(handle, bool) { return false; }
  static handle cast(const orthodr_py::List& value, return_value_policy, handle) {
    return value.object().release();
  }
};
template<> struct type_caster<orthodr_py::NumericVector> {
  PYBIND11_TYPE_CASTER(orthodr_py::NumericVector, const_name("numpy.ndarray"));
  bool load(handle value, bool) { this->value = orthodr_py::NumericVector(value); return true; }
  static handle cast(const orthodr_py::NumericVector& value, return_value_policy, handle) {
    return orthodr_py::pack(value.vector()).release();
  }
};
template<> struct type_caster<orthodr_py::Nullable<orthodr_py::NumericVector>> {
  using OptionalVector = orthodr_py::Nullable<orthodr_py::NumericVector>;
  PYBIND11_TYPE_CASTER(OptionalVector, const_name("numpy.ndarray | None"));
  bool load(handle value, bool) { this->value = OptionalVector(value); return true; }
};
} // namespace pybind11::detail

// Original kernels use printf only for the optional OpenMP availability notice.
inline void orthodr_printf(const char* format, ...) {
  va_list arguments; va_start(arguments, format);
  std::vfprintf(stderr, format, arguments); va_end(arguments);
}
