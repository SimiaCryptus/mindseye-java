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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class StaticScalarLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StaticScalarLossLayer.class);
  private double target = 0.0;

  public StaticScalarLossLayer() {
  }

  protected StaticScalarLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public double getTarget() {
    return target;
  }

  @Nonnull
  public StaticScalarLossLayer setTarget(final double target) {
    this.target = target;
    return this;
  }

  @SuppressWarnings("unused")
  public static StaticScalarLossLayer fromJson(@Nonnull final JsonObject json,
                                               com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new StaticScalarLossLayer(json);
  }

  public static @SuppressWarnings("unused")
  StaticScalarLossLayer[] addRefs(StaticScalarLossLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StaticScalarLossLayer::addRef)
        .toArray((x) -> new StaticScalarLossLayer[x]);
  }

  public static @SuppressWarnings("unused")
  StaticScalarLossLayer[][] addRefs(StaticScalarLossLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(StaticScalarLossLayer::addRefs)
        .toArray((x) -> new StaticScalarLossLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length)
      throw new IllegalArgumentException();
    //if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    final Result in0 = inObj[0];
    TensorList indata = in0.getData();
    return new Result(new TensorArray(
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
          @Nullable final Tensor a = indata.get(dataIndex);
          final double diff = Math.abs(a.get(0) - getTarget());
          return new Tensor(new double[]{diff}, 1);
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (in0.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
              @Nullable final Tensor a = indata.get(dataIndex);
              Tensor tensor = data.get(dataIndex);
              final double deriv = tensor.get(0) * (a.get(0) - getTarget() < 0 ? -1 : 1);
              return new Tensor(new double[]{deriv}, 1);
            }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return in0.isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  StaticScalarLossLayer addRef() {
    return (StaticScalarLossLayer) super.addRef();
  }
}
