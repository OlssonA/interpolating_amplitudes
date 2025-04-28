module     p2_gg_httbar_d26h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d26h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd26
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
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(108) :: acd26
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd26(1)=dotproduct(k2,ninjaE3)
      acd26(2)=abb26(14)
      acd26(3)=dotproduct(ninjaA0,ninjaE3)
      acd26(4)=abb26(17)
      acd26(5)=dotproduct(ninjaE3,spvae2k2)
      acd26(6)=abb26(11)
      acd26(7)=dotproduct(ninjaE3,spvak2e1)
      acd26(8)=abb26(12)
      acd26(9)=dotproduct(ninjaE3,spvak2l3)
      acd26(10)=abb26(15)
      acd26(11)=dotproduct(ninjaE3,spval5k2)
      acd26(12)=abb26(16)
      acd26(13)=dotproduct(ninjaE3,spvak1l3)
      acd26(14)=abb26(18)
      acd26(15)=dotproduct(ninjaE3,spvak2e2)
      acd26(16)=abb26(19)
      acd26(17)=dotproduct(ninjaE3,spval3k2)
      acd26(18)=abb26(20)
      acd26(19)=dotproduct(ninjaE3,spval4l3)
      acd26(20)=abb26(21)
      acd26(21)=dotproduct(ninjaE3,spvae2e1)
      acd26(22)=abb26(22)
      acd26(23)=dotproduct(ninjaE3,spvae1l5)
      acd26(24)=abb26(23)
      acd26(25)=dotproduct(ninjaE3,spval4k2)
      acd26(26)=abb26(24)
      acd26(27)=dotproduct(ninjaE3,spvak2k1)
      acd26(28)=abb26(25)
      acd26(29)=dotproduct(ninjaE3,spval5e1)
      acd26(30)=abb26(26)
      acd26(31)=dotproduct(ninjaE3,spval3k1)
      acd26(32)=abb26(27)
      acd26(33)=dotproduct(ninjaE3,spvak2l4)
      acd26(34)=abb26(28)
      acd26(35)=dotproduct(ninjaE3,spval5k1)
      acd26(36)=abb26(29)
      acd26(37)=dotproduct(ninjaE3,spvae2k1)
      acd26(38)=abb26(30)
      acd26(39)=dotproduct(ninjaE3,spvae2l4)
      acd26(40)=abb26(31)
      acd26(41)=dotproduct(ninjaE3,spvak1e2)
      acd26(42)=abb26(32)
      acd26(43)=dotproduct(ninjaE3,spvak1l5)
      acd26(44)=abb26(33)
      acd26(45)=dotproduct(ninjaE3,spval4l5)
      acd26(46)=abb26(34)
      acd26(47)=dotproduct(ninjaE3,spval4e2)
      acd26(48)=abb26(35)
      acd26(49)=dotproduct(ninjaE3,spvak2l5)
      acd26(50)=abb26(36)
      acd26(51)=dotproduct(ninjaE3,spvak1k2)
      acd26(52)=abb26(37)
      acd26(53)=dotproduct(ninjaE3,spvae2l3)
      acd26(54)=abb26(39)
      acd26(55)=dotproduct(ninjaE3,spval5l4)
      acd26(56)=abb26(40)
      acd26(57)=dotproduct(ninjaE3,spval3e2)
      acd26(58)=abb26(42)
      acd26(59)=dotproduct(ninjaE3,spval5l3)
      acd26(60)=abb26(43)
      acd26(61)=dotproduct(ninjaE3,spval3l5)
      acd26(62)=abb26(44)
      acd26(63)=dotproduct(ninjaE3,spval3l4)
      acd26(64)=abb26(45)
      acd26(65)=dotproduct(ninjaE3,spvae1e2)
      acd26(66)=abb26(52)
      acd26(67)=dotproduct(ninjaE3,spvae1l3)
      acd26(68)=abb26(57)
      acd26(69)=dotproduct(ninjaE3,spvae1k2)
      acd26(70)=abb26(72)
      acd26(71)=dotproduct(ninjaE3,spval3e1)
      acd26(72)=abb26(73)
      acd26(73)=acd26(2)*acd26(1)
      acd26(74)=acd26(4)*acd26(3)
      acd26(75)=acd26(6)*acd26(5)
      acd26(76)=acd26(8)*acd26(7)
      acd26(77)=acd26(10)*acd26(9)
      acd26(78)=acd26(12)*acd26(11)
      acd26(79)=acd26(14)*acd26(13)
      acd26(80)=acd26(16)*acd26(15)
      acd26(81)=acd26(18)*acd26(17)
      acd26(82)=acd26(20)*acd26(19)
      acd26(83)=acd26(22)*acd26(21)
      acd26(84)=acd26(24)*acd26(23)
      acd26(85)=acd26(26)*acd26(25)
      acd26(86)=acd26(28)*acd26(27)
      acd26(87)=acd26(30)*acd26(29)
      acd26(88)=acd26(32)*acd26(31)
      acd26(89)=acd26(34)*acd26(33)
      acd26(90)=acd26(36)*acd26(35)
      acd26(91)=acd26(38)*acd26(37)
      acd26(92)=acd26(40)*acd26(39)
      acd26(93)=acd26(42)*acd26(41)
      acd26(94)=acd26(44)*acd26(43)
      acd26(95)=acd26(46)*acd26(45)
      acd26(96)=acd26(48)*acd26(47)
      acd26(97)=acd26(50)*acd26(49)
      acd26(98)=acd26(52)*acd26(51)
      acd26(99)=acd26(54)*acd26(53)
      acd26(100)=acd26(56)*acd26(55)
      acd26(101)=acd26(58)*acd26(57)
      acd26(102)=acd26(60)*acd26(59)
      acd26(103)=acd26(62)*acd26(61)
      acd26(104)=acd26(64)*acd26(63)
      acd26(105)=-acd26(66)*acd26(65)
      acd26(106)=acd26(68)*acd26(67)
      acd26(107)=acd26(70)*acd26(69)
      acd26(108)=-acd26(72)*acd26(71)
      acd26(73)=acd26(108)+acd26(107)+acd26(106)+acd26(105)+acd26(104)+acd26(10&
      &3)+acd26(102)+acd26(101)+acd26(100)+acd26(99)+acd26(98)+acd26(97)+acd26(&
      &96)+acd26(95)+acd26(94)+acd26(93)+acd26(92)+acd26(91)+acd26(90)+acd26(89&
      &)+acd26(88)+acd26(87)+acd26(86)+acd26(85)+acd26(84)+acd26(83)+acd26(82)+&
      &acd26(81)+acd26(80)+acd26(79)+acd26(78)+acd26(77)+acd26(76)+acd26(75)+ac&
      &d26(73)+2.0_ki*acd26(74)
      brack(ninjaidxt0x0mu0)=acd26(73)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d26h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
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
end module     p2_gg_httbar_d26h4l132_qp
