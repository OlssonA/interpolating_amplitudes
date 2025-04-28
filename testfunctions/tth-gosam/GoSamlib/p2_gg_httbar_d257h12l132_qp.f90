module     p2_gg_httbar_d257h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d257h12l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd257h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd257
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd257h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd257
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd257(1)=dotproduct(ninjaA0,ninjaE3)
      acd257(2)=dotproduct(ninjaE3,spvae2l5)
      acd257(3)=dotproduct(ninjaE3,spvae1e2)
      acd257(4)=abb257(8)
      acd257(5)=dotproduct(ninjaE3,spvae2l4)
      acd257(6)=abb257(46)
      acd257(7)=dotproduct(ninjaE3,spvak2e1)
      acd257(8)=abb257(19)
      acd257(9)=dotproduct(ninjaE3,spval3l5)
      acd257(10)=dotproduct(ninjaE3,spvae2l3)
      acd257(11)=dotproduct(ninjaE3,spval3l4)
      acd257(12)=dotproduct(ninjaE3,spvak2l5)
      acd257(13)=dotproduct(ninjaE3,spvae2k2)
      acd257(14)=abb257(22)
      acd257(15)=dotproduct(ninjaE3,spvak2l4)
      acd257(16)=abb257(31)
      acd257(17)=dotproduct(ninjaE3,spvak2e2)
      acd257(18)=dotproduct(ninjaE3,spvae1l4)
      acd257(19)=dotproduct(ninjaE3,spvae2e1)
      acd257(20)=abb257(27)
      acd257(21)=-acd257(6)*acd257(11)
      acd257(22)=acd257(4)*acd257(9)
      acd257(21)=acd257(22)+acd257(21)
      acd257(21)=acd257(10)*acd257(21)
      acd257(22)=acd257(15)*acd257(16)
      acd257(23)=acd257(12)*acd257(14)
      acd257(22)=acd257(22)+acd257(23)
      acd257(22)=acd257(13)*acd257(22)
      acd257(23)=acd257(6)*acd257(5)
      acd257(24)=-acd257(2)*acd257(4)
      acd257(23)=acd257(23)+acd257(24)
      acd257(23)=acd257(1)*acd257(23)
      acd257(24)=acd257(2)*acd257(7)*acd257(8)
      acd257(21)=2.0_ki*acd257(23)+acd257(24)+acd257(22)+acd257(21)
      acd257(21)=acd257(3)*acd257(21)
      acd257(22)=-acd257(17)*acd257(18)*acd257(19)*acd257(20)
      acd257(21)=acd257(22)+acd257(21)
      brack(ninjaidxt0x0mu0)=acd257(21)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d257h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd257h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d257h12l132_qp
