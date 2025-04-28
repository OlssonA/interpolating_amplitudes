module     p2_gg_httbar_d10h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d10h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(22) :: acd10
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd10(1)=dotproduct(k2,ninjaE3)
      acd10(2)=abb10(10)
      acd10(3)=dotproduct(ninjaE3,spval3k2)
      acd10(4)=abb10(30)
      acd10(5)=dotproduct(ninjaE3,spvak1k2)
      acd10(6)=dotproduct(ninjaE3,spvak2k1)
      acd10(7)=abb10(9)
      acd10(8)=dotproduct(ninjaE3,spval3k1)
      acd10(9)=abb10(23)
      acd10(10)=dotproduct(ninjaE3,spvak1l4)
      acd10(11)=dotproduct(ninjaE3,spval5k1)
      acd10(12)=abb10(19)
      acd10(13)=dotproduct(ninjaE3,spvak1l3)
      acd10(14)=abb10(24)
      acd10(15)=dotproduct(ninjaE3,spvak2l4)
      acd10(16)=dotproduct(ninjaE3,spval5k2)
      acd10(17)=dotproduct(ninjaE3,spvak2l3)
      acd10(18)=acd10(14)*acd10(17)
      acd10(19)=acd10(12)*acd10(15)
      acd10(18)=acd10(18)+acd10(19)
      acd10(18)=acd10(16)*acd10(18)
      acd10(19)=-acd10(14)*acd10(13)
      acd10(20)=-acd10(12)*acd10(10)
      acd10(19)=acd10(19)+acd10(20)
      acd10(19)=acd10(11)*acd10(19)
      acd10(20)=acd10(8)*acd10(9)
      acd10(21)=acd10(6)*acd10(7)
      acd10(20)=acd10(20)+acd10(21)
      acd10(20)=acd10(5)*acd10(20)
      acd10(21)=-acd10(3)*acd10(4)
      acd10(22)=acd10(1)*acd10(2)
      acd10(21)=acd10(21)+acd10(22)
      acd10(21)=acd10(1)*acd10(21)
      acd10(18)=acd10(21)+acd10(20)+acd10(19)+acd10(18)
      brack(ninjaidxt1x0mu0)=acd10(18)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(73) :: acd10
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd10(1)=dotproduct(k2,ninjaA1)
      acd10(2)=dotproduct(k2,ninjaE3)
      acd10(3)=abb10(10)
      acd10(4)=dotproduct(ninjaE3,spval3k2)
      acd10(5)=abb10(30)
      acd10(6)=dotproduct(ninjaA1,spval3k2)
      acd10(7)=dotproduct(ninjaA1,spvak1k2)
      acd10(8)=dotproduct(ninjaE3,spvak2k1)
      acd10(9)=abb10(9)
      acd10(10)=dotproduct(ninjaE3,spval3k1)
      acd10(11)=abb10(23)
      acd10(12)=dotproduct(ninjaA1,spvak2k1)
      acd10(13)=dotproduct(ninjaE3,spvak1k2)
      acd10(14)=dotproduct(ninjaA1,spval3k1)
      acd10(15)=dotproduct(ninjaA1,spval5k2)
      acd10(16)=dotproduct(ninjaE3,spvak2l4)
      acd10(17)=abb10(19)
      acd10(18)=dotproduct(ninjaE3,spvak2l3)
      acd10(19)=abb10(24)
      acd10(20)=dotproduct(ninjaA1,spvak1l3)
      acd10(21)=dotproduct(ninjaE3,spval5k1)
      acd10(22)=dotproduct(ninjaA1,spval5k1)
      acd10(23)=dotproduct(ninjaE3,spvak1l3)
      acd10(24)=dotproduct(ninjaE3,spvak1l4)
      acd10(25)=dotproduct(ninjaA1,spvak1l4)
      acd10(26)=dotproduct(ninjaA1,spvak2l4)
      acd10(27)=dotproduct(ninjaE3,spval5k2)
      acd10(28)=dotproduct(ninjaA1,spvak2l3)
      acd10(29)=dotproduct(k2,ninjaA0)
      acd10(30)=dotproduct(ninjaA0,spval3k2)
      acd10(31)=abb10(11)
      acd10(32)=dotproduct(ninjaA0,spvak1k2)
      acd10(33)=dotproduct(ninjaA0,spvak2k1)
      acd10(34)=dotproduct(ninjaA0,spval3k1)
      acd10(35)=dotproduct(ninjaA0,spval5k2)
      acd10(36)=dotproduct(ninjaA0,spvak1l3)
      acd10(37)=dotproduct(ninjaA0,spval5k1)
      acd10(38)=dotproduct(ninjaA0,spvak1l4)
      acd10(39)=dotproduct(ninjaA0,spvak2l4)
      acd10(40)=dotproduct(ninjaA0,spvak2l3)
      acd10(41)=abb10(13)
      acd10(42)=abb10(17)
      acd10(43)=abb10(12)
      acd10(44)=abb10(14)
      acd10(45)=abb10(15)
      acd10(46)=abb10(16)
      acd10(47)=abb10(18)
      acd10(48)=abb10(27)
      acd10(49)=abb10(22)
      acd10(50)=abb10(21)
      acd10(51)=acd10(4)*acd10(5)
      acd10(52)=acd10(3)*acd10(2)
      acd10(51)=-acd10(51)+2.0_ki*acd10(52)
      acd10(52)=acd10(1)*acd10(51)
      acd10(53)=acd10(8)*acd10(9)
      acd10(54)=acd10(10)*acd10(11)
      acd10(53)=acd10(53)+acd10(54)
      acd10(54)=acd10(7)*acd10(53)
      acd10(55)=acd10(16)*acd10(17)
      acd10(56)=acd10(18)*acd10(19)
      acd10(55)=acd10(55)+acd10(56)
      acd10(56)=acd10(15)*acd10(55)
      acd10(57)=acd10(23)*acd10(19)
      acd10(58)=acd10(24)*acd10(17)
      acd10(57)=acd10(57)+acd10(58)
      acd10(58)=-acd10(22)*acd10(57)
      acd10(59)=acd10(5)*acd10(2)
      acd10(60)=-acd10(6)*acd10(59)
      acd10(61)=acd10(9)*acd10(13)
      acd10(62)=acd10(12)*acd10(61)
      acd10(63)=acd10(11)*acd10(13)
      acd10(64)=acd10(14)*acd10(63)
      acd10(65)=acd10(21)*acd10(19)
      acd10(66)=-acd10(20)*acd10(65)
      acd10(67)=acd10(21)*acd10(17)
      acd10(68)=-acd10(25)*acd10(67)
      acd10(69)=acd10(27)*acd10(17)
      acd10(70)=acd10(26)*acd10(69)
      acd10(71)=acd10(27)*acd10(19)
      acd10(72)=acd10(28)*acd10(71)
      acd10(52)=acd10(72)+acd10(70)+acd10(68)+acd10(66)+acd10(64)+acd10(62)+acd&
      &10(60)+acd10(58)+acd10(56)+acd10(54)+acd10(52)
      acd10(51)=acd10(29)*acd10(51)
      acd10(53)=acd10(32)*acd10(53)
      acd10(54)=acd10(35)*acd10(55)
      acd10(55)=-acd10(37)*acd10(57)
      acd10(56)=-acd10(30)*acd10(59)
      acd10(57)=acd10(31)*acd10(2)
      acd10(58)=acd10(33)*acd10(61)
      acd10(59)=acd10(34)*acd10(63)
      acd10(60)=-acd10(36)*acd10(65)
      acd10(61)=-acd10(38)*acd10(67)
      acd10(62)=acd10(39)*acd10(69)
      acd10(63)=acd10(40)*acd10(71)
      acd10(64)=acd10(41)*acd10(13)
      acd10(65)=acd10(42)*acd10(8)
      acd10(66)=acd10(43)*acd10(10)
      acd10(67)=acd10(44)*acd10(27)
      acd10(68)=acd10(45)*acd10(23)
      acd10(69)=acd10(46)*acd10(4)
      acd10(70)=acd10(47)*acd10(21)
      acd10(71)=acd10(48)*acd10(24)
      acd10(72)=acd10(49)*acd10(16)
      acd10(73)=acd10(50)*acd10(18)
      acd10(51)=acd10(73)+acd10(72)+acd10(71)+acd10(70)+acd10(69)+acd10(68)+acd&
      &10(67)+acd10(66)+acd10(65)+acd10(64)+acd10(63)+acd10(62)+acd10(61)+acd10&
      &(60)+acd10(59)+acd10(58)+acd10(57)+acd10(56)+acd10(55)+acd10(54)+acd10(5&
      &1)+acd10(53)
      brack(ninjaidxt0x0mu0)=acd10(51)
      brack(ninjaidxt0x1mu0)=acd10(52)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d10h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
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
end module     p2_gg_httbar_d10h4l132_qp
