module     p2_gg_httbar_d66h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d66h12l131_qp.f90
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
      use p2_gg_httbar_abbrevd66h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd66
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(99) :: acd66
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd66(1)=dotproduct(k1,ninjaE3)
      acd66(2)=abb66(38)
      acd66(3)=dotproduct(k2,ninjaE3)
      acd66(4)=dotproduct(ninjaE3,spvak2l4)
      acd66(5)=abb66(12)
      acd66(6)=dotproduct(ninjaE3,spvak2l5)
      acd66(7)=abb66(21)
      acd66(8)=dotproduct(ninjaE3,spvak1l5)
      acd66(9)=abb66(27)
      acd66(10)=dotproduct(ninjaE3,spvak1l4)
      acd66(11)=abb66(29)
      acd66(12)=dotproduct(ninjaA,ninjaE3)
      acd66(13)=abb66(28)
      acd66(14)=abb66(61)
      acd66(15)=dotproduct(ninjaE3,spval3l5)
      acd66(16)=abb66(22)
      acd66(17)=dotproduct(ninjaE3,spval3l4)
      acd66(18)=abb66(74)
      acd66(19)=dotproduct(ninjaE3,spvak2k1)
      acd66(20)=abb66(23)
      acd66(21)=abb66(19)
      acd66(22)=dotproduct(ninjaE3,spvak2l3)
      acd66(23)=dotproduct(ninjaE3,spval3k1)
      acd66(24)=abb66(14)
      acd66(25)=dotproduct(ninjaE3,spvak1l3)
      acd66(26)=dotproduct(ninjaE3,spvak1k2)
      acd66(27)=abb66(16)
      acd66(28)=dotproduct(k1,ninjaA)
      acd66(29)=dotproduct(ninjaA,ninjaA)
      acd66(30)=dotproduct(ninjaA,spvak2l4)
      acd66(31)=dotproduct(ninjaA,spvak2l5)
      acd66(32)=dotproduct(ninjaA,spval3l5)
      acd66(33)=dotproduct(ninjaA,spval3l4)
      acd66(34)=abb66(44)
      acd66(35)=dotproduct(k2,ninjaA)
      acd66(36)=dotproduct(ninjaA,spvak2k1)
      acd66(37)=abb66(49)
      acd66(38)=dotproduct(l3,ninjaE3)
      acd66(39)=abb66(33)
      acd66(40)=dotproduct(ninjaA,spvak1l5)
      acd66(41)=dotproduct(ninjaA,spvak1l4)
      acd66(42)=abb66(32)
      acd66(43)=dotproduct(ninjaA,spvak2l3)
      acd66(44)=dotproduct(ninjaA,spval3k1)
      acd66(45)=dotproduct(ninjaA,spvak1l3)
      acd66(46)=dotproduct(ninjaA,spvak1k2)
      acd66(47)=abb66(9)
      acd66(48)=abb66(10)
      acd66(49)=abb66(11)
      acd66(50)=abb66(39)
      acd66(51)=abb66(15)
      acd66(52)=abb66(25)
      acd66(53)=abb66(18)
      acd66(54)=abb66(17)
      acd66(55)=dotproduct(ninjaE3,spval5l3)
      acd66(56)=abb66(20)
      acd66(57)=abb66(24)
      acd66(58)=abb66(26)
      acd66(59)=abb66(42)
      acd66(60)=dotproduct(ninjaE3,spval3k2)
      acd66(61)=abb66(48)
      acd66(62)=dotproduct(ninjaE3,spval5k2)
      acd66(63)=abb66(66)
      acd66(64)=dotproduct(ninjaE3,spval5k1)
      acd66(65)=abb66(67)
      acd66(66)=dotproduct(ninjaE3,spval4k1)
      acd66(67)=abb66(69)
      acd66(68)=acd66(5)*acd66(4)
      acd66(69)=acd66(8)*acd66(9)
      acd66(70)=acd66(11)*acd66(10)
      acd66(69)=acd66(70)+acd66(68)+acd66(69)
      acd66(71)=acd66(1)-acd66(3)
      acd66(72)=acd66(71)*acd66(2)
      acd66(73)=acd66(7)*acd66(6)
      acd66(72)=acd66(72)-acd66(73)
      acd66(73)=acd66(72)+acd66(69)
      acd66(74)=acd66(27)*acd66(26)
      acd66(75)=-acd66(14)*acd66(1)
      acd66(76)=acd66(21)*acd66(3)
      acd66(75)=acd66(74)+acd66(76)+acd66(75)
      acd66(75)=acd66(6)*acd66(75)
      acd66(76)=2.0_ki*acd66(12)
      acd66(69)=acd66(76)*acd66(69)
      acd66(77)=acd66(13)*acd66(4)
      acd66(78)=acd66(16)*acd66(15)
      acd66(79)=acd66(17)*acd66(18)
      acd66(77)=-acd66(79)+acd66(77)+acd66(78)
      acd66(78)=-acd66(71)*acd66(77)
      acd66(79)=acd66(7)*acd66(15)
      acd66(80)=acd66(24)*acd66(23)
      acd66(79)=acd66(79)-acd66(80)
      acd66(80)=-acd66(22)*acd66(79)
      acd66(81)=acd66(12)*acd66(72)
      acd66(82)=acd66(20)*acd66(19)
      acd66(83)=acd66(3)*acd66(82)
      acd66(84)=acd66(25)*acd66(9)*acd66(15)
      acd66(69)=acd66(84)+acd66(83)+2.0_ki*acd66(81)+acd66(78)+acd66(69)+acd66(&
      &75)+acd66(80)
      acd66(68)=acd66(68)+acd66(70)+acd66(72)
      acd66(70)=ninjaP+acd66(29)
      acd66(68)=acd66(70)*acd66(68)
      acd66(72)=acd66(76)*acd66(2)
      acd66(72)=acd66(72)-acd66(77)
      acd66(75)=acd66(82)-acd66(72)
      acd66(75)=acd66(35)*acd66(75)
      acd66(77)=-acd66(31)*acd66(7)
      acd66(78)=acd66(40)*acd66(9)
      acd66(80)=acd66(41)*acd66(11)
      acd66(77)=acd66(42)+acd66(80)+acd66(78)+acd66(77)
      acd66(77)=acd66(76)*acd66(77)
      acd66(72)=acd66(28)*acd66(72)
      acd66(78)=-acd66(16)*acd66(71)
      acd66(80)=-acd66(22)*acd66(7)
      acd66(78)=acd66(78)+acd66(80)
      acd66(78)=acd66(32)*acd66(78)
      acd66(80)=-acd66(28)*acd66(6)
      acd66(81)=-acd66(31)*acd66(1)
      acd66(80)=acd66(80)+acd66(81)
      acd66(80)=acd66(14)*acd66(80)
      acd66(81)=acd66(35)*acd66(6)
      acd66(82)=acd66(31)*acd66(3)
      acd66(81)=acd66(81)+acd66(82)
      acd66(81)=acd66(21)*acd66(81)
      acd66(82)=-acd66(13)*acd66(71)
      acd66(76)=acd66(5)*acd66(76)
      acd66(76)=acd66(82)+acd66(76)
      acd66(76)=acd66(30)*acd66(76)
      acd66(79)=-acd66(43)*acd66(79)
      acd66(82)=acd66(46)*acd66(27)
      acd66(82)=acd66(53)+acd66(82)
      acd66(82)=acd66(6)*acd66(82)
      acd66(83)=acd66(36)*acd66(20)
      acd66(83)=acd66(37)+acd66(83)
      acd66(83)=acd66(3)*acd66(83)
      acd66(84)=acd66(45)*acd66(9)
      acd66(84)=acd66(57)+acd66(84)
      acd66(84)=acd66(15)*acd66(84)
      acd66(70)=acd66(9)*acd66(70)
      acd66(70)=acd66(54)+acd66(70)
      acd66(70)=acd66(8)*acd66(70)
      acd66(85)=acd66(44)*acd66(24)
      acd66(85)=acd66(49)+acd66(85)
      acd66(85)=acd66(22)*acd66(85)
      acd66(86)=acd66(32)*acd66(9)
      acd66(86)=acd66(51)+acd66(86)
      acd66(86)=acd66(25)*acd66(86)
      acd66(74)=acd66(31)*acd66(74)
      acd66(71)=acd66(33)*acd66(18)*acd66(71)
      acd66(87)=acd66(34)*acd66(1)
      acd66(88)=acd66(39)*acd66(38)
      acd66(89)=acd66(47)*acd66(19)
      acd66(90)=acd66(48)*acd66(4)
      acd66(91)=acd66(50)*acd66(23)
      acd66(92)=acd66(52)*acd66(26)
      acd66(93)=acd66(56)*acd66(55)
      acd66(94)=acd66(58)*acd66(10)
      acd66(95)=acd66(59)*acd66(17)
      acd66(96)=acd66(61)*acd66(60)
      acd66(97)=acd66(63)*acd66(62)
      acd66(98)=acd66(65)*acd66(64)
      acd66(99)=acd66(67)*acd66(66)
      acd66(68)=acd66(99)+acd66(98)+acd66(97)+acd66(96)+acd66(95)+acd66(94)+acd&
      &66(93)+acd66(92)+acd66(91)+acd66(90)+acd66(89)+acd66(88)+acd66(87)+acd66&
      &(79)+acd66(71)+acd66(76)+acd66(74)+acd66(81)+acd66(80)+acd66(78)+acd66(7&
      &2)+acd66(77)+acd66(75)+acd66(86)+acd66(85)+acd66(70)+acd66(84)+acd66(83)&
      &+acd66(68)+acd66(82)
      brack(ninjaidxt1mu0)=acd66(69)
      brack(ninjaidxt0mu0)=acd66(68)
      brack(ninjaidxt0mu2)=acd66(73)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d66h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd66h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d66h12l131_qp
