module     p2_gg_httbar_d11h8l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d11h8l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd11h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(12) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(k1,ninjaE3)
      acd11(2)=dotproduct(k2,ninjaE3)
      acd11(3)=abb11(34)
      acd11(4)=dotproduct(ninjaE3,spval4l5)
      acd11(5)=abb11(16)
      acd11(6)=dotproduct(ninjaE3,spval3l5)
      acd11(7)=abb11(19)
      acd11(8)=dotproduct(ninjaE3,spvak2l3)
      acd11(9)=abb11(25)
      acd11(10)=acd11(8)*acd11(9)
      acd11(11)=acd11(6)*acd11(7)
      acd11(12)=acd11(4)*acd11(5)
      acd11(10)=acd11(12)+acd11(10)-acd11(11)
      acd11(11)=-acd11(1)*acd11(10)
      acd11(12)=acd11(2)-acd11(1)
      acd11(12)=acd11(3)*acd11(12)
      acd11(10)=acd11(12)+acd11(10)
      acd11(10)=acd11(2)*acd11(10)
      acd11(10)=acd11(11)+acd11(10)
      brack(ninjaidxt2mu0)=acd11(10)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd11h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(89) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(k1,ninjaE3)
      acd11(2)=dotproduct(k2,ninjaE4)
      acd11(3)=abb11(34)
      acd11(4)=dotproduct(ninjaE4,spval4l5)
      acd11(5)=abb11(16)
      acd11(6)=dotproduct(ninjaE4,spval3l5)
      acd11(7)=abb11(19)
      acd11(8)=dotproduct(ninjaE4,spvak2l3)
      acd11(9)=abb11(25)
      acd11(10)=dotproduct(k1,ninjaE4)
      acd11(11)=dotproduct(k2,ninjaE3)
      acd11(12)=dotproduct(ninjaE3,spval4l5)
      acd11(13)=dotproduct(ninjaE3,spval3l5)
      acd11(14)=dotproduct(ninjaE3,spvak2l3)
      acd11(15)=abb11(15)
      acd11(16)=dotproduct(k1,ninjaA)
      acd11(17)=dotproduct(k2,ninjaA)
      acd11(18)=dotproduct(ninjaA,spval4l5)
      acd11(19)=dotproduct(ninjaA,spval3l5)
      acd11(20)=dotproduct(ninjaA,spvak2l3)
      acd11(21)=abb11(11)
      acd11(22)=abb11(9)
      acd11(23)=dotproduct(l5,ninjaE3)
      acd11(24)=abb11(17)
      acd11(25)=dotproduct(ninjaA,ninjaE3)
      acd11(26)=dotproduct(ninjaE3,spvak2l5)
      acd11(27)=abb11(10)
      acd11(28)=dotproduct(ninjaE3,spvak1k2)
      acd11(29)=abb11(12)
      acd11(30)=dotproduct(ninjaE3,spvak1l5)
      acd11(31)=abb11(13)
      acd11(32)=abb11(20)
      acd11(33)=dotproduct(ninjaE3,spvak1l3)
      acd11(34)=abb11(18)
      acd11(35)=abb11(39)
      acd11(36)=dotproduct(ninjaE3,spval3k2)
      acd11(37)=abb11(21)
      acd11(38)=dotproduct(ninjaE3,spval5l3)
      acd11(39)=abb11(22)
      acd11(40)=dotproduct(ninjaE3,spval3k1)
      acd11(41)=abb11(23)
      acd11(42)=abb11(28)
      acd11(43)=dotproduct(ninjaE3,spvak2k1)
      acd11(44)=abb11(26)
      acd11(45)=dotproduct(ninjaE3,spval4k1)
      acd11(46)=abb11(30)
      acd11(47)=dotproduct(ninjaE3,spval5k2)
      acd11(48)=abb11(36)
      acd11(49)=dotproduct(ninjaE3,spval4k2)
      acd11(50)=abb11(38)
      acd11(51)=dotproduct(l5,ninjaA)
      acd11(52)=dotproduct(ninjaA,ninjaA)
      acd11(53)=dotproduct(ninjaA,spvak2l5)
      acd11(54)=dotproduct(ninjaA,spvak1k2)
      acd11(55)=dotproduct(ninjaA,spvak1l5)
      acd11(56)=dotproduct(ninjaA,spvak1l3)
      acd11(57)=dotproduct(ninjaA,spval3k2)
      acd11(58)=dotproduct(ninjaA,spval5l3)
      acd11(59)=dotproduct(ninjaA,spval3k1)
      acd11(60)=dotproduct(ninjaA,spvak2k1)
      acd11(61)=dotproduct(ninjaA,spval4k1)
      acd11(62)=dotproduct(ninjaA,spval5k2)
      acd11(63)=dotproduct(ninjaA,spval4k2)
      acd11(64)=abb11(14)
      acd11(65)=-acd11(10)+2.0_ki*acd11(2)
      acd11(65)=acd11(65)*acd11(3)
      acd11(66)=acd11(9)*acd11(8)
      acd11(67)=acd11(7)*acd11(6)
      acd11(68)=acd11(5)*acd11(4)
      acd11(65)=acd11(68)+acd11(65)+acd11(66)-acd11(67)
      acd11(65)=acd11(65)*acd11(11)
      acd11(66)=acd11(2)-acd11(10)
      acd11(67)=-acd11(12)*acd11(66)
      acd11(68)=acd11(1)*acd11(4)
      acd11(67)=acd11(67)+acd11(68)
      acd11(67)=acd11(67)*acd11(5)
      acd11(68)=-acd11(13)*acd11(66)
      acd11(69)=acd11(1)*acd11(6)
      acd11(68)=acd11(68)+acd11(69)
      acd11(68)=acd11(68)*acd11(7)
      acd11(66)=-acd11(14)*acd11(66)
      acd11(69)=acd11(1)*acd11(8)
      acd11(66)=acd11(66)+acd11(69)
      acd11(66)=acd11(66)*acd11(9)
      acd11(69)=acd11(2)*acd11(1)*acd11(3)
      acd11(65)=-acd11(65)+acd11(67)-acd11(68)+acd11(66)+acd11(69)-acd11(15)
      acd11(66)=acd11(9)*acd11(20)
      acd11(67)=acd11(7)*acd11(19)
      acd11(68)=acd11(5)*acd11(18)
      acd11(66)=acd11(68)+acd11(66)-acd11(67)
      acd11(67)=-acd11(16)+2.0_ki*acd11(17)
      acd11(67)=acd11(3)*acd11(67)
      acd11(67)=acd11(22)+acd11(67)+acd11(66)
      acd11(67)=acd11(11)*acd11(67)
      acd11(68)=acd11(17)-acd11(16)
      acd11(69)=acd11(14)*acd11(68)
      acd11(70)=-acd11(1)*acd11(20)
      acd11(69)=acd11(70)+acd11(69)
      acd11(69)=acd11(9)*acd11(69)
      acd11(70)=-acd11(13)*acd11(68)
      acd11(71)=acd11(1)*acd11(19)
      acd11(70)=acd11(71)+acd11(70)
      acd11(70)=acd11(7)*acd11(70)
      acd11(71)=acd11(12)*acd11(68)
      acd11(72)=-acd11(1)*acd11(18)
      acd11(71)=acd11(72)+acd11(71)
      acd11(71)=acd11(5)*acd11(71)
      acd11(72)=acd11(50)*acd11(49)
      acd11(73)=acd11(48)*acd11(47)
      acd11(74)=acd11(46)*acd11(45)
      acd11(75)=acd11(44)*acd11(43)
      acd11(76)=acd11(41)*acd11(40)
      acd11(77)=acd11(39)*acd11(38)
      acd11(78)=acd11(37)*acd11(36)
      acd11(79)=acd11(34)*acd11(33)
      acd11(80)=acd11(31)*acd11(30)
      acd11(81)=acd11(29)*acd11(28)
      acd11(82)=acd11(27)*acd11(26)
      acd11(83)=acd11(24)*acd11(23)
      acd11(84)=acd11(15)*acd11(25)
      acd11(85)=acd11(14)*acd11(42)
      acd11(86)=acd11(13)*acd11(35)
      acd11(87)=acd11(12)*acd11(32)
      acd11(88)=acd11(3)*acd11(17)
      acd11(89)=acd11(21)-acd11(88)
      acd11(89)=acd11(1)*acd11(89)
      acd11(67)=acd11(67)+acd11(71)+acd11(70)+acd11(69)+acd11(89)+acd11(87)+acd&
      &11(86)+acd11(85)+2.0_ki*acd11(84)+acd11(83)+acd11(82)+acd11(81)+acd11(80&
      &)+acd11(79)+acd11(78)+acd11(77)+acd11(76)+acd11(75)+acd11(74)+acd11(72)+&
      &acd11(73)
      acd11(69)=-ninjaP*acd11(65)
      acd11(66)=acd11(88)+acd11(66)
      acd11(66)=acd11(68)*acd11(66)
      acd11(68)=acd11(50)*acd11(63)
      acd11(70)=acd11(48)*acd11(62)
      acd11(71)=acd11(46)*acd11(61)
      acd11(72)=acd11(44)*acd11(60)
      acd11(73)=acd11(41)*acd11(59)
      acd11(74)=acd11(39)*acd11(58)
      acd11(75)=acd11(37)*acd11(57)
      acd11(76)=acd11(34)*acd11(56)
      acd11(77)=acd11(31)*acd11(55)
      acd11(78)=acd11(29)*acd11(54)
      acd11(79)=acd11(27)*acd11(53)
      acd11(80)=acd11(24)*acd11(51)
      acd11(81)=acd11(15)*acd11(52)
      acd11(82)=acd11(20)*acd11(42)
      acd11(83)=acd11(19)*acd11(35)
      acd11(84)=acd11(18)*acd11(32)
      acd11(85)=acd11(16)*acd11(21)
      acd11(86)=acd11(17)*acd11(22)
      acd11(66)=acd11(86)+acd11(85)+acd11(84)+acd11(83)+acd11(82)+acd11(81)+acd&
      &11(80)+acd11(79)+acd11(78)+acd11(77)+acd11(76)+acd11(75)+acd11(74)+acd11&
      &(73)+acd11(72)+acd11(71)+acd11(70)+acd11(64)+acd11(68)+acd11(69)+acd11(6&
      &6)
      brack(ninjaidxt1mu0)=acd11(67)
      brack(ninjaidxt0mu0)=acd11(66)
      brack(ninjaidxt0mu2)=-acd11(65)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d11h8_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd11h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = - a(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d11h8l131_qp
