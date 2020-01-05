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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class MeanSqLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);

  public MeanSqLossLayer() {
  }

  protected MeanSqLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static MeanSqLossLayer fromJson(@Nonnull final JsonObject json,
                                         Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json);
  }

  public static @SuppressWarnings("unused")
  MeanSqLossLayer[] addRefs(MeanSqLossLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayer::addRef)
        .toArray((x) -> new MeanSqLossLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MeanSqLossLayer[][] addRefs(MeanSqLossLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MeanSqLossLayer::addRefs)
        .toArray((x) -> new MeanSqLossLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (2 != inObj.length)
      throw new IllegalArgumentException();
    final int leftLength = inObj[0].getData().length();
    final int rightLength = inObj[1].getData().length();
    if (leftLength != rightLength && leftLength != 1 && rightLength != 1) {
      throw new IllegalArgumentException(leftLength + " != " + rightLength);
    }
    @Nonnull final Tensor diffs[] = new Tensor[leftLength];
    return new Result(
        new TensorArray(RefIntStream.range(0, leftLength).mapToObj(dataIndex -> {
          @Nullable final Tensor a = inObj[0].getData().get(1 == leftLength ? 0 : dataIndex);
          @Nullable final Tensor b = inObj[1].getData().get(1 == rightLength ? 0 : dataIndex);
          if (a.length() != b.length()) {
            throw new IllegalArgumentException(
                String.format("%s != %s", RefArrays.toString(a.getDimensions()),
                    RefArrays.toString(b.getDimensions())));
          }
          @Nonnull final Tensor r = a.minus(b);
          diffs[dataIndex] = r;
          return new Tensor(new double[]{r.sumSq() / r.length()}, 1);
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        RefStream<Tensor> tensorStream = RefIntStream
            .range(0, data.length()).parallel().mapToObj(dataIndex -> {
              @Nullable
              Tensor tensor = data.get(dataIndex);
              Tensor diff = diffs[dataIndex];
              return diff.scale(tensor.get(0) * 2.0 / diff.length());
            }).collect(RefCollectors.toList()).stream();
        if (1 == leftLength) {
          tensorStream = RefStream.of(tensorStream.reduce((a, b) -> {
            return a.addAndFree(b);
          }).get());
        }
        @Nonnull final TensorList array = new TensorArray(tensorStream.toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, array);
      }
      if (inObj[1].isAlive()) {
        RefStream<Tensor> tensorStream = RefIntStream
            .range(0, data.length()).parallel().mapToObj(dataIndex -> {
              @Nullable
              Tensor tensor = data.get(dataIndex);
              return diffs[dataIndex].scale(tensor.get(0) * 2.0 / diffs[dataIndex].length());
            }).collect(RefCollectors.toList()).stream();
        if (1 == rightLength) {
          tensorStream = RefStream.of(tensorStream.reduce((a, b) -> {
            return a.addAndFree(b);
          }).get());
        }
        @Nonnull final TensorList array = new TensorArray(tensorStream.map(x -> {
          return x.scale(-1);
        }).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, array);
      }
    }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[1].isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MeanSqLossLayer addRef() {
    return (MeanSqLossLayer) super.addRef();
  }
}
