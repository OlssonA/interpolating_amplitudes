module     p2_gg_httbar_d78h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d78h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd78h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd78
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
      use p2_gg_httbar_abbrevd78h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(37) :: acd78
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd78(1)=dotproduct(ninjaA0,ninjaE3)
      acd78(2)=dotproduct(ninjaE3,spvae1k2)
      acd78(3)=abb78(31)
      acd78(4)=dotproduct(ninjaE3,spval5e1)
      acd78(5)=abb78(30)
      acd78(6)=dotproduct(ninjaE3,spvak2e1)
      acd78(7)=abb78(20)
      acd78(8)=dotproduct(ninjaE3,spvae2e1)
      acd78(9)=abb78(29)
      acd78(10)=dotproduct(ninjaE3,spval4e1)
      acd78(11)=abb78(23)
      acd78(12)=dotproduct(ninjaE3,spvae1e2)
      acd78(13)=abb78(40)
      acd78(14)=dotproduct(ninjaE3,spvae1l5)
      acd78(15)=abb78(38)
      acd78(16)=dotproduct(ninjaE3,spval3e1)
      acd78(17)=abb78(9)
      acd78(18)=abb78(11)
      acd78(19)=abb78(15)
      acd78(20)=abb78(13)
      acd78(21)=abb78(16)
      acd78(22)=abb78(37)
      acd78(23)=abb78(41)
      acd78(24)=dotproduct(ninjaE3,spvae1l3)
      acd78(25)=abb78(48)
      acd78(26)=abb78(12)
      acd78(27)=abb78(45)
      acd78(28)=abb78(32)
      acd78(29)=abb78(44)
      acd78(30)=abb78(47)
      acd78(31)=acd78(3)*acd78(2)
      acd78(32)=acd78(5)*acd78(4)
      acd78(33)=acd78(7)*acd78(6)
      acd78(34)=acd78(9)*acd78(8)
      acd78(35)=-acd78(11)*acd78(10)
      acd78(36)=acd78(13)*acd78(12)
      acd78(37)=acd78(15)*acd78(14)
      acd78(31)=acd78(37)+acd78(36)+acd78(35)+acd78(34)+acd78(33)+acd78(31)+acd&
      &78(32)
      acd78(31)=acd78(1)*acd78(31)
      acd78(32)=acd78(17)*acd78(16)
      acd78(33)=acd78(18)*acd78(4)
      acd78(34)=acd78(19)*acd78(6)
      acd78(35)=acd78(20)*acd78(8)
      acd78(36)=acd78(21)*acd78(10)
      acd78(32)=acd78(36)+acd78(35)+acd78(34)+acd78(33)+acd78(32)
      acd78(32)=acd78(2)*acd78(32)
      acd78(33)=acd78(25)*acd78(4)
      acd78(34)=acd78(26)*acd78(6)
      acd78(35)=acd78(27)*acd78(8)
      acd78(36)=acd78(28)*acd78(10)
      acd78(33)=acd78(36)+acd78(35)+acd78(34)+acd78(33)
      acd78(33)=acd78(24)*acd78(33)
      acd78(34)=acd78(29)*acd78(12)
      acd78(35)=acd78(30)*acd78(14)
      acd78(34)=acd78(35)+acd78(34)
      acd78(34)=acd78(10)*acd78(34)
      acd78(35)=-acd78(22)*acd78(12)
      acd78(36)=-acd78(23)*acd78(14)
      acd78(35)=acd78(36)+acd78(35)
      acd78(35)=acd78(16)*acd78(35)
      acd78(31)=2.0_ki*acd78(31)+acd78(32)+acd78(33)+acd78(35)+acd78(34)
      brack(ninjaidxt0x0mu0)=acd78(31)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d78h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd78h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d78h0l132_qp
