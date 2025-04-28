module     p2_gg_httbar_d75h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d75h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd75
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(99) :: acd75
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd75(1)=dotproduct(ninjaE3,spvak2e1)
      acd75(2)=abb75(21)
      acd75(3)=dotproduct(ninjaE3,spvae1k2)
      acd75(4)=abb75(18)
      acd75(5)=dotproduct(ninjaE3,spvae2e1)
      acd75(6)=abb75(43)
      acd75(7)=dotproduct(ninjaE3,spvae1e2)
      acd75(8)=abb75(41)
      acd75(9)=dotproduct(ninjaE3,spval5e1)
      acd75(10)=abb75(27)
      acd75(11)=dotproduct(ninjaE3,spval4e1)
      acd75(12)=abb75(28)
      acd75(13)=dotproduct(ninjaE3,spvae1l4)
      acd75(14)=abb75(29)
      acd75(15)=dotproduct(k2,ninjaE3)
      acd75(16)=abb75(30)
      acd75(17)=abb75(14)
      acd75(18)=dotproduct(ninjaA,ninjaE3)
      acd75(19)=dotproduct(ninjaE3,spval5k2)
      acd75(20)=abb75(9)
      acd75(21)=dotproduct(ninjaE3,spval4k2)
      acd75(22)=abb75(11)
      acd75(23)=abb75(35)
      acd75(24)=dotproduct(ninjaE3,spvae2k2)
      acd75(25)=abb75(31)
      acd75(26)=dotproduct(ninjaE3,spval3e1)
      acd75(27)=dotproduct(ninjaE3,spvak2l3)
      acd75(28)=dotproduct(ninjaE3,spval5l3)
      acd75(29)=dotproduct(ninjaE3,spval4l3)
      acd75(30)=dotproduct(ninjaE3,spvae2l3)
      acd75(31)=abb75(15)
      acd75(32)=dotproduct(ninjaE3,spvak2l4)
      acd75(33)=abb75(20)
      acd75(34)=dotproduct(ninjaE3,spvak2e2)
      acd75(35)=abb75(24)
      acd75(36)=abb75(25)
      acd75(37)=dotproduct(ninjaE3,spval3k2)
      acd75(38)=dotproduct(ninjaE3,spvae1l3)
      acd75(39)=dotproduct(ninjaE3,spval3l4)
      acd75(40)=dotproduct(ninjaE3,spval3e2)
      acd75(41)=abb75(23)
      acd75(42)=abb75(37)
      acd75(43)=dotproduct(k2,ninjaA)
      acd75(44)=dotproduct(ninjaA,spvak2e1)
      acd75(45)=dotproduct(ninjaA,spvae1k2)
      acd75(46)=dotproduct(ninjaA,ninjaA)
      acd75(47)=dotproduct(ninjaA,spvae2e1)
      acd75(48)=dotproduct(ninjaA,spvae1e2)
      acd75(49)=dotproduct(ninjaA,spval5e1)
      acd75(50)=dotproduct(ninjaA,spval4e1)
      acd75(51)=dotproduct(ninjaA,spvae1l4)
      acd75(52)=abb75(33)
      acd75(53)=dotproduct(ninjaA,spval5k2)
      acd75(54)=dotproduct(ninjaA,spval3e1)
      acd75(55)=dotproduct(ninjaA,spval4k2)
      acd75(56)=dotproduct(ninjaA,spval3k2)
      acd75(57)=dotproduct(ninjaA,spvae1l3)
      acd75(58)=dotproduct(ninjaA,spvak2l4)
      acd75(59)=dotproduct(ninjaA,spvak2l3)
      acd75(60)=dotproduct(ninjaA,spvak2e2)
      acd75(61)=dotproduct(ninjaA,spval5l3)
      acd75(62)=dotproduct(ninjaA,spval4l3)
      acd75(63)=dotproduct(ninjaA,spval3l4)
      acd75(64)=dotproduct(ninjaA,spvae2k2)
      acd75(65)=dotproduct(ninjaA,spval3e2)
      acd75(66)=dotproduct(ninjaA,spvae2l3)
      acd75(67)=abb75(16)
      acd75(68)=abb75(10)
      acd75(69)=abb75(12)
      acd75(70)=abb75(17)
      acd75(71)=abb75(22)
      acd75(72)=abb75(19)
      acd75(73)=abb75(36)
      acd75(74)=abb75(26)
      acd75(75)=abb75(32)
      acd75(76)=acd75(2)*acd75(1)
      acd75(77)=acd75(4)*acd75(3)
      acd75(78)=acd75(10)*acd75(9)
      acd75(79)=acd75(5)*acd75(6)
      acd75(80)=acd75(7)*acd75(8)
      acd75(81)=acd75(11)*acd75(12)
      acd75(82)=acd75(13)*acd75(14)
      acd75(76)=acd75(78)+acd75(76)+acd75(77)-acd75(79)-acd75(80)+acd75(81)-acd&
      &75(82)
      acd75(77)=-acd75(18)*acd75(76)
      acd75(78)=acd75(17)*acd75(15)
      acd75(79)=acd75(31)*acd75(5)
      acd75(80)=acd75(36)*acd75(11)
      acd75(81)=acd75(32)*acd75(33)
      acd75(82)=acd75(34)*acd75(35)
      acd75(78)=acd75(78)-acd75(80)+acd75(81)+acd75(79)+acd75(82)
      acd75(79)=acd75(23)*acd75(1)
      acd75(79)=acd75(79)+acd75(78)
      acd75(79)=acd75(3)*acd75(79)
      acd75(80)=acd75(16)*acd75(15)
      acd75(81)=acd75(19)*acd75(20)
      acd75(82)=acd75(21)*acd75(22)
      acd75(83)=acd75(24)*acd75(25)
      acd75(80)=acd75(83)+acd75(80)+acd75(81)+acd75(82)
      acd75(81)=acd75(1)*acd75(80)
      acd75(82)=acd75(27)*acd75(2)
      acd75(83)=acd75(28)*acd75(10)
      acd75(84)=acd75(29)*acd75(12)
      acd75(85)=acd75(30)*acd75(6)
      acd75(82)=acd75(82)+acd75(83)+acd75(84)-acd75(85)
      acd75(83)=acd75(26)*acd75(82)
      acd75(84)=acd75(37)*acd75(4)
      acd75(85)=acd75(39)*acd75(14)
      acd75(86)=acd75(40)*acd75(8)
      acd75(84)=-acd75(86)+acd75(84)-acd75(85)
      acd75(85)=acd75(38)*acd75(84)
      acd75(86)=acd75(41)*acd75(7)
      acd75(87)=acd75(42)*acd75(13)
      acd75(86)=acd75(86)-acd75(87)
      acd75(87)=-acd75(9)*acd75(86)
      acd75(77)=2.0_ki*acd75(77)+acd75(79)+acd75(83)+acd75(81)+acd75(85)+acd75(&
      &87)
      acd75(79)=-ninjaP-acd75(46)
      acd75(79)=acd75(76)*acd75(79)
      acd75(81)=2.0_ki*acd75(18)
      acd75(83)=-acd75(4)*acd75(81)
      acd75(78)=acd75(83)+acd75(78)
      acd75(78)=acd75(45)*acd75(78)
      acd75(82)=acd75(54)*acd75(82)
      acd75(83)=acd75(59)*acd75(2)
      acd75(85)=acd75(61)*acd75(10)
      acd75(87)=acd75(62)*acd75(12)
      acd75(88)=-acd75(66)*acd75(6)
      acd75(83)=acd75(68)+acd75(88)+acd75(87)+acd75(85)+acd75(83)
      acd75(83)=acd75(26)*acd75(83)
      acd75(85)=-acd75(2)*acd75(81)
      acd75(80)=acd75(85)+acd75(80)
      acd75(80)=acd75(44)*acd75(80)
      acd75(84)=acd75(57)*acd75(84)
      acd75(85)=acd75(53)*acd75(20)
      acd75(87)=acd75(55)*acd75(22)
      acd75(88)=acd75(64)*acd75(25)
      acd75(85)=acd75(67)+acd75(88)+acd75(87)+acd75(85)
      acd75(85)=acd75(1)*acd75(85)
      acd75(87)=acd75(56)*acd75(4)
      acd75(88)=-acd75(63)*acd75(14)
      acd75(89)=-acd75(65)*acd75(8)
      acd75(87)=acd75(71)+acd75(89)+acd75(88)+acd75(87)
      acd75(87)=acd75(38)*acd75(87)
      acd75(88)=acd75(58)*acd75(33)
      acd75(89)=acd75(60)*acd75(35)
      acd75(88)=acd75(69)+acd75(89)+acd75(88)
      acd75(88)=acd75(3)*acd75(88)
      acd75(89)=acd75(45)*acd75(1)
      acd75(90)=acd75(44)*acd75(3)
      acd75(89)=acd75(89)+acd75(90)
      acd75(89)=acd75(23)*acd75(89)
      acd75(90)=-acd75(10)*acd75(81)
      acd75(86)=acd75(90)-acd75(86)
      acd75(86)=acd75(49)*acd75(86)
      acd75(90)=acd75(16)*acd75(1)
      acd75(91)=acd75(17)*acd75(3)
      acd75(90)=acd75(90)+acd75(91)
      acd75(90)=acd75(43)*acd75(90)
      acd75(91)=acd75(6)*acd75(81)
      acd75(92)=acd75(31)*acd75(3)
      acd75(91)=acd75(91)+acd75(92)
      acd75(91)=acd75(47)*acd75(91)
      acd75(92)=acd75(8)*acd75(81)
      acd75(93)=-acd75(41)*acd75(9)
      acd75(92)=acd75(92)+acd75(93)
      acd75(92)=acd75(48)*acd75(92)
      acd75(93)=-acd75(12)*acd75(81)
      acd75(94)=-acd75(36)*acd75(3)
      acd75(93)=acd75(93)+acd75(94)
      acd75(93)=acd75(50)*acd75(93)
      acd75(94)=acd75(14)*acd75(81)
      acd75(95)=acd75(42)*acd75(9)
      acd75(94)=acd75(94)+acd75(95)
      acd75(94)=acd75(51)*acd75(94)
      acd75(81)=acd75(52)*acd75(81)
      acd75(95)=acd75(70)*acd75(5)
      acd75(96)=acd75(72)*acd75(7)
      acd75(97)=acd75(73)*acd75(9)
      acd75(98)=acd75(74)*acd75(11)
      acd75(99)=acd75(75)*acd75(13)
      acd75(78)=acd75(99)+acd75(98)+acd75(97)+acd75(96)+acd75(95)+acd75(81)+acd&
      &75(94)+acd75(93)+acd75(92)+acd75(91)+acd75(90)+acd75(86)+acd75(89)+acd75&
      &(78)+acd75(80)+acd75(83)+acd75(82)+acd75(87)+acd75(85)+acd75(84)+acd75(8&
      &8)+acd75(79)
      brack(ninjaidxt1mu0)=acd75(77)
      brack(ninjaidxt0mu0)=acd75(78)
      brack(ninjaidxt0mu2)=-acd75(76)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d75h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd75h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d75h0l131
