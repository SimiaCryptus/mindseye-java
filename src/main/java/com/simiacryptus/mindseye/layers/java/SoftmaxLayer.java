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
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class SoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxLayer.class);
  double maxInput = 50;

  public SoftmaxLayer() {
  }

  protected SoftmaxLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static SoftmaxLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SoftmaxLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int itemCnt = inObj[0].getData().length();
    @Nonnull
    final double[] sumA = new double[itemCnt];
    @Nonnull
    final Tensor expA[] = new Tensor[itemCnt];
    final Tensor[] outputA = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nullable
      final Tensor input = inObj[0].getData().get(dataIndex);
      assert 1 < input.length() : "input.length() = " + input.length();

      @Nullable
      final Tensor exp;
      final DoubleSummaryStatistics summaryStatistics = com.simiacryptus.ref.wrappers.RefDoubleStream
          .of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
      final double max = summaryStatistics.getMax();
      //final double min = summaryStatistics.getMin();
      exp = input.map(x -> {
        double xx = Math.exp(x - max);
        return Double.isFinite(xx) ? xx : 0;
      });
      assert com.simiacryptus.ref.wrappers.RefArrays.stream(exp.getData()).allMatch(Double::isFinite);
      assert com.simiacryptus.ref.wrappers.RefArrays.stream(exp.getData()).allMatch(v -> v >= 0);
      //assert exp.sum() > 0;
      final double sum = 0 < exp.sum() ? exp.sum() : 1;
      assert Double.isFinite(sum);
      expA[dataIndex] = exp;
      sumA[dataIndex] = sum;
      return exp.map(x -> x / sum);
    }).toArray(i -> new Tensor[i]);
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(outputA)
        .flatMapToDouble(x -> com.simiacryptus.ref.wrappers.RefArrays.stream(x.getData()))
        .allMatch(v -> Double.isFinite(v));
    return new Result(new TensorArray(outputA),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          if (inObj[0].isAlive()) {
            final Tensor[] passbackA = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt)
                .mapToObj(dataIndex -> {
                  Tensor deltaTensor = data.get(dataIndex);
                  @Nullable
                  final double[] delta = deltaTensor.getData();
                  @Nullable
                  final double[] expdata = expA[dataIndex].getData();
                  @Nonnull
                  final Tensor passback = new Tensor(data.getDimensions());
                  final int dim = expdata.length;
                  double dot = 0;
                  for (int i = 0; i < expdata.length; i++) {
                    dot += delta[i] * expdata[i];
                  }
                  final double sum = sumA[dataIndex];
                  for (int i = 0; i < dim; i++) {
                    double value = 0;
                    value = (sum * delta[i] - dot) * expdata[i] / (sum * sum);
                    passback.set(i, value);
                  }
                  return passback;
                }).toArray(i -> new Tensor[i]);
            assert com.simiacryptus.ref.wrappers.RefArrays.stream(passbackA)
                .flatMapToDouble(x -> com.simiacryptus.ref.wrappers.RefArrays.stream(x.getData()))
                .allMatch(v -> Double.isFinite(v));
            @Nonnull
            TensorArray tensorArray = new TensorArray(passbackA);
            inObj[0].accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") SoftmaxLayer addRef() {
    return (SoftmaxLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") SoftmaxLayer[] addRefs(SoftmaxLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayer::addRef)
        .toArray((x) -> new SoftmaxLayer[x]);
  }

  public static @SuppressWarnings("unused") SoftmaxLayer[][] addRefs(SoftmaxLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SoftmaxLayer::addRefs)
        .toArray((x) -> new SoftmaxLayer[x][]);
  }
}
