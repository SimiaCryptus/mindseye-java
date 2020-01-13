/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase {

  public SumInputsLayer() {
  }

  protected SumInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }

  public static PipelineNetwork combine(PipelineNetwork... networks) {
    PipelineNetwork temp_55_0005 = PipelineNetwork.combine(new SumInputsLayer(), PipelineNetwork.addRefs(networks));
    if (null != networks)
      ReferenceCounting.freeRefs(networks);
    return temp_55_0005;
  }

  public static @SuppressWarnings("unused") SumInputsLayer[] addRefs(SumInputsLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayer::addRef)
        .toArray((x) -> new SumInputsLayer[x]);
  }

  public static @SuppressWarnings("unused") SumInputsLayer[][] addRefs(SumInputsLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumInputsLayer::addRefs)
        .toArray((x) -> new SumInputsLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    try {
      return new Result(RefUtil.get(RefArrays.stream(Result.addRefs(inObj)).parallel().map(x -> {
        TensorList temp_55_0001 = x.getData();
        if (null != x)
          x.freeRef();
        return temp_55_0001;
      }).reduce((l, r) -> {
        assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
        TensorArray temp_55_0002 = new TensorArray(RefIntStream.range(0, l.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
              @Nullable
              final Tensor left = l.get(1 == l.length() ? 0 : i);
              @Nullable
              final Tensor right = r.get(1 == r.length() ? 0 : i);
              @Nullable
              Tensor tensor;
              if (right.length() == 1) {
                tensor = left.mapParallel(RefUtil.wrapInterface((DoubleUnaryOperator) v -> v + right.get(0),
                    right == null ? null : right.addRef()));
              } else {
                tensor = left.reduceParallel(right == null ? null : right.addRef(), (v1, v2) -> v1 + v2);
              }
              if (null != right)
                right.freeRef();
              if (null != left)
                left.freeRef();
              return tensor;
            }, r == null ? null : r.addRef(), l == null ? null : l.addRef())).toArray(i -> new Tensor[i]));
        if (null != r)
          r.freeRef();
        if (null != l)
          l.freeRef();
        return temp_55_0002;
      })), new Result.Accumulator() {
        {
          Result.addRefs(inObj);
        }

        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList delta) {
          for (@Nonnull
          final Result input : inObj) {
            if (input.isAlive()) {
              @Nonnull
              TensorList projectedDelta = delta == null ? null : delta.addRef();
              TensorList temp_55_0007 = input.getData();
              if (1 < projectedDelta.length() && temp_55_0007.length() == 1) {
                projectedDelta = new TensorArray(RefUtil.get(projectedDelta.stream().parallel().reduce((a, b) -> {
                  Tensor temp_55_0003 = a.addAndFree(b == null ? null : b.addRef());
                  if (null != b)
                    b.freeRef();
                  if (null != a)
                    a.freeRef();
                  return temp_55_0003;
                })));
              }
              if (null != temp_55_0007)
                temp_55_0007.freeRef();
              TensorList temp_55_0008 = input.getData();
              if (1 < Tensor.length(projectedDelta.getDimensions())
                  && Tensor.length(temp_55_0008.getDimensions()) == 1) {
                @Nonnull
                TensorArray new_projectedDelta = new TensorArray(projectedDelta.stream().map(t -> {
                  Tensor temp_55_0004 = new Tensor(new double[] { t.sum() });
                  if (null != t)
                    t.freeRef();
                  return temp_55_0004;
                }).toArray(i -> new Tensor[i]));
                projectedDelta = new_projectedDelta == null ? null : new_projectedDelta;
              }
              if (null != temp_55_0008)
                temp_55_0008.freeRef();
              input.accumulate(buffer == null ? null : buffer.addRef(), projectedDelta == null ? null : projectedDelta);
            }
          }
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused") void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      }) {

        {
          Result.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          for (@Nonnull
          final Result element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
        }

      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") SumInputsLayer addRef() {
    return (SumInputsLayer) super.addRef();
  }

}
