module     p2_gg_httbar_d70h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d70h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd70h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd70
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
      use p2_gg_httbar_abbrevd70h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(47) :: acd70
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd70(1)=dotproduct(ninjaA0,ninjaE3)
      acd70(2)=dotproduct(ninjaE3,spvae2l5)
      acd70(3)=abb70(9)
      acd70(4)=dotproduct(ninjaE3,spvak2e2)
      acd70(5)=abb70(26)
      acd70(6)=dotproduct(ninjaE3,spvae2k2)
      acd70(7)=abb70(52)
      acd70(8)=dotproduct(ninjaE3,spvak1e2)
      acd70(9)=abb70(33)
      acd70(10)=dotproduct(ninjaE3,spvae2e1)
      acd70(11)=abb70(45)
      acd70(12)=dotproduct(ninjaE3,spvae2k1)
      acd70(13)=abb70(20)
      acd70(14)=dotproduct(ninjaE3,spval4e2)
      acd70(15)=abb70(56)
      acd70(16)=dotproduct(ninjaE3,spvae2l4)
      acd70(17)=abb70(54)
      acd70(18)=dotproduct(ninjaE3,spvae1e2)
      acd70(19)=abb70(49)
      acd70(20)=abb70(14)
      acd70(21)=abb70(35)
      acd70(22)=abb70(21)
      acd70(23)=dotproduct(ninjaE3,spval3e2)
      acd70(24)=abb70(22)
      acd70(25)=abb70(34)
      acd70(26)=abb70(10)
      acd70(27)=dotproduct(ninjaE3,spvae2l3)
      acd70(28)=abb70(28)
      acd70(29)=abb70(19)
      acd70(30)=abb70(24)
      acd70(31)=abb70(32)
      acd70(32)=abb70(23)
      acd70(33)=abb70(13)
      acd70(34)=abb70(58)
      acd70(35)=abb70(48)
      acd70(36)=abb70(39)
      acd70(37)=abb70(31)
      acd70(38)=abb70(59)
      acd70(39)=acd70(3)*acd70(2)
      acd70(40)=acd70(5)*acd70(4)
      acd70(41)=acd70(7)*acd70(6)
      acd70(42)=acd70(9)*acd70(8)
      acd70(43)=acd70(11)*acd70(10)
      acd70(44)=acd70(13)*acd70(12)
      acd70(45)=acd70(15)*acd70(14)
      acd70(46)=acd70(17)*acd70(16)
      acd70(47)=acd70(19)*acd70(18)
      acd70(39)=acd70(47)+acd70(46)+acd70(45)+acd70(44)+acd70(43)+acd70(42)+acd&
      &70(41)+acd70(39)+acd70(40)
      acd70(39)=acd70(1)*acd70(39)
      acd70(40)=acd70(20)*acd70(2)
      acd70(41)=acd70(26)*acd70(6)
      acd70(42)=acd70(28)*acd70(27)
      acd70(43)=acd70(29)*acd70(10)
      acd70(44)=acd70(30)*acd70(12)
      acd70(45)=acd70(31)*acd70(16)
      acd70(40)=acd70(45)+acd70(44)+acd70(43)+acd70(42)+acd70(41)+acd70(40)
      acd70(40)=acd70(4)*acd70(40)
      acd70(41)=acd70(24)*acd70(2)
      acd70(42)=acd70(32)*acd70(6)
      acd70(43)=acd70(36)*acd70(10)
      acd70(44)=acd70(37)*acd70(12)
      acd70(45)=acd70(38)*acd70(16)
      acd70(41)=acd70(45)+acd70(44)+acd70(43)+acd70(42)+acd70(41)
      acd70(41)=acd70(23)*acd70(41)
      acd70(42)=acd70(21)*acd70(8)
      acd70(43)=acd70(22)*acd70(14)
      acd70(44)=acd70(25)*acd70(18)
      acd70(42)=acd70(44)+acd70(43)+acd70(42)
      acd70(42)=acd70(2)*acd70(42)
      acd70(43)=acd70(33)*acd70(8)
      acd70(44)=acd70(34)*acd70(14)
      acd70(45)=acd70(35)*acd70(18)
      acd70(43)=acd70(45)+acd70(44)+acd70(43)
      acd70(43)=acd70(27)*acd70(43)
      acd70(39)=2.0_ki*acd70(39)+acd70(40)+acd70(41)+acd70(43)+acd70(42)
      brack(ninjaidxt0x0mu0)=acd70(39)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d70h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd70h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
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
end module     p2_gg_httbar_d70h12l132_qp
