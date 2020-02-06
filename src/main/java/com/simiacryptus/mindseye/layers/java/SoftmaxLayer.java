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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.DoubleSummaryStatistics;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class SoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxLayer.class);
  double maxInput = 50;

  public SoftmaxLayer() {
  }

  protected SoftmaxLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_08_0008 = inObj[0].getData();
    final int itemCnt = temp_08_0008.length();
    temp_08_0008.freeRef();
    @Nonnull final double[] sumA = new double[itemCnt];
    @Nonnull final Tensor expA[] = new Tensor[itemCnt];
    final Tensor[] outputA = RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          TensorList temp_08_0009 = inObj[0].getData();
          @Nullable final Tensor input = temp_08_0009.get(dataIndex);
          temp_08_0009.freeRef();
          assert 1 < input.length() : "input.length() = " + input.length();

          final DoubleSummaryStatistics summaryStatistics = RefDoubleStream.of(input.getData())
              .filter(Double::isFinite).summaryStatistics();
          final double max = summaryStatistics.getMax();
          //final double min = summaryStatistics.getMin();
          @Nullable final Tensor exp = input.map(x -> {
            double xx = Math.exp(x - max);
            return Double.isFinite(xx) ? xx : 0;
          });
          input.freeRef();
          assert RefArrays.stream(exp.getData()).allMatch(Double::isFinite);
          assert RefArrays.stream(exp.getData()).allMatch(v -> v >= 0);
          //assert exp.sum() > 0;
          final double sum = 0 < exp.sum() ? exp.sum() : 1;
          assert Double.isFinite(sum);
          sumA[dataIndex] = sum;
          Tensor temp_08_0003 = exp.map(x -> x / sum);
          RefUtil.set(expA, dataIndex, exp);
          return temp_08_0003;
        }, RefUtil.addRefs(expA), RefUtil.addRefs(inObj))).toArray(Tensor[]::new);
    assert RefArrays.stream(RefUtil.addRefs(outputA)).flatMapToDouble(x -> {
      RefDoubleStream temp_08_0005 = RefArrays.stream(x.getData());
      x.freeRef();
      return temp_08_0005;
    }).allMatch(Double::isFinite);
    try {
      return new Result(new TensorArray(RefUtil.addRefs(outputA)), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          RefUtil.addRefs(expA);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (inObj[0].isAlive()) {
            final Tensor[] passbackA = RefIntStream.range(0, itemCnt)
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor deltaTensor = data.get(dataIndex);
                  @Nullable final double[] delta = deltaTensor.getData();
                  deltaTensor.freeRef();
                  @Nullable final double[] expdata = expA[dataIndex].getData();
                  @Nonnull final Tensor passback = new Tensor(data.getDimensions());
                  final int dim = expdata.length;
                  double dot = 0;
                  for (int i = 0; i < expdata.length; i++) {
                    dot += delta[i] * expdata[i];
                  }
                  final double sum = sumA[dataIndex];
                  for (int i = 0; i < dim; i++) {
                    double value = (sum * delta[i] - dot) * expdata[i] / (sum * sum);
                    passback.set(i, value);
                  }
                  return passback;
                }, RefUtil.addRefs(expA), data.addRef())).toArray(Tensor[]::new);
            assert RefArrays.stream(RefUtil.addRefs(passbackA)).flatMapToDouble(x -> {
              RefDoubleStream temp_08_0006 = RefArrays.stream(x.getData());
              x.freeRef();
              return temp_08_0006;
            }).allMatch(Double::isFinite);
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefUtil.addRefs(passbackA));
            RefUtil.freeRef(passbackA);
            inObj[0].accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(inObj);
          RefUtil.freeRef(expA);
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return inObj[0].isAlive();
        }

        public void _free() {
          RefUtil.freeRef(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(inObj);
      RefUtil.freeRef(outputA);
      RefUtil.freeRef(expA);
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

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SoftmaxLayer addRef() {
    return (SoftmaxLayer) super.addRef();
  }
}
