module     p2_gg_httbar_d72h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d72h4l131_qp.f90
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
      use p2_gg_httbar_abbrevd72h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd72
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd72h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(88) :: acd72
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd72(1)=dotproduct(ninjaE3,spvak2e2)
      acd72(2)=abb72(17)
      acd72(3)=dotproduct(ninjaE3,spvae2l4)
      acd72(4)=abb72(25)
      acd72(5)=dotproduct(ninjaE3,spvae2l5)
      acd72(6)=abb72(54)
      acd72(7)=dotproduct(ninjaE3,spvae2k1)
      acd72(8)=abb72(26)
      acd72(9)=dotproduct(ninjaE3,spvae1e2)
      acd72(10)=abb72(48)
      acd72(11)=dotproduct(ninjaE3,spvak1e2)
      acd72(12)=abb72(32)
      acd72(13)=dotproduct(ninjaE3,spvae2e1)
      acd72(14)=abb72(40)
      acd72(15)=dotproduct(ninjaE3,spval5e2)
      acd72(16)=abb72(47)
      acd72(17)=dotproduct(ninjaA,ninjaE3)
      acd72(18)=abb72(9)
      acd72(19)=abb72(10)
      acd72(20)=dotproduct(ninjaE3,spvae2l3)
      acd72(21)=abb72(11)
      acd72(22)=dotproduct(ninjaE3,spvae2k2)
      acd72(23)=abb72(12)
      acd72(24)=abb72(31)
      acd72(25)=abb72(22)
      acd72(26)=abb72(29)
      acd72(27)=abb72(33)
      acd72(28)=dotproduct(ninjaE3,spval3e2)
      acd72(29)=abb72(66)
      acd72(30)=abb72(39)
      acd72(31)=abb72(65)
      acd72(32)=abb72(55)
      acd72(33)=abb72(20)
      acd72(34)=abb72(63)
      acd72(35)=abb72(30)
      acd72(36)=abb72(64)
      acd72(37)=dotproduct(ninjaA,ninjaA)
      acd72(38)=dotproduct(ninjaA,spvak2e2)
      acd72(39)=dotproduct(ninjaA,spvae2l4)
      acd72(40)=dotproduct(ninjaA,spvae2l5)
      acd72(41)=dotproduct(ninjaA,spvae2k1)
      acd72(42)=dotproduct(ninjaA,spvae1e2)
      acd72(43)=dotproduct(ninjaA,spvak1e2)
      acd72(44)=dotproduct(ninjaA,spvae2e1)
      acd72(45)=dotproduct(ninjaA,spval5e2)
      acd72(46)=abb72(27)
      acd72(47)=dotproduct(ninjaA,spvae2l3)
      acd72(48)=dotproduct(ninjaA,spvae2k2)
      acd72(49)=dotproduct(ninjaA,spval3e2)
      acd72(50)=abb72(15)
      acd72(51)=abb72(19)
      acd72(52)=abb72(53)
      acd72(53)=abb72(18)
      acd72(54)=abb72(13)
      acd72(55)=abb72(16)
      acd72(56)=abb72(23)
      acd72(57)=abb72(21)
      acd72(58)=abb72(28)
      acd72(59)=abb72(24)
      acd72(60)=acd72(6)*acd72(5)
      acd72(61)=acd72(8)*acd72(7)
      acd72(62)=acd72(10)*acd72(9)
      acd72(63)=acd72(12)*acd72(11)
      acd72(64)=acd72(14)*acd72(13)
      acd72(65)=acd72(16)*acd72(15)
      acd72(60)=acd72(60)+acd72(61)-acd72(62)+acd72(63)+acd72(64)+acd72(65)
      acd72(61)=acd72(2)*acd72(1)
      acd72(62)=acd72(4)*acd72(3)
      acd72(61)=acd72(61)+acd72(62)+acd72(60)
      acd72(62)=acd72(17)*acd72(61)
      acd72(63)=acd72(19)*acd72(5)
      acd72(64)=acd72(24)*acd72(7)
      acd72(65)=acd72(25)*acd72(13)
      acd72(66)=acd72(22)*acd72(23)
      acd72(63)=acd72(66)+acd72(63)+acd72(64)+acd72(65)
      acd72(64)=acd72(18)*acd72(3)
      acd72(65)=acd72(21)*acd72(20)
      acd72(64)=acd72(65)+acd72(64)+acd72(63)
      acd72(64)=acd72(1)*acd72(64)
      acd72(65)=acd72(26)*acd72(9)
      acd72(66)=acd72(27)*acd72(11)
      acd72(67)=acd72(30)*acd72(15)
      acd72(65)=-acd72(67)+acd72(65)+acd72(66)
      acd72(66)=acd72(29)*acd72(28)
      acd72(66)=acd72(66)+acd72(65)
      acd72(66)=acd72(3)*acd72(66)
      acd72(67)=acd72(31)*acd72(28)
      acd72(68)=-acd72(5)*acd72(67)
      acd72(69)=acd72(32)*acd72(20)
      acd72(70)=acd72(9)*acd72(69)
      acd72(71)=acd72(33)*acd72(20)
      acd72(72)=acd72(11)*acd72(71)
      acd72(73)=acd72(34)*acd72(20)
      acd72(74)=acd72(15)*acd72(73)
      acd72(75)=acd72(35)*acd72(28)
      acd72(76)=acd72(7)*acd72(75)
      acd72(77)=acd72(36)*acd72(28)
      acd72(78)=acd72(13)*acd72(77)
      acd72(62)=acd72(78)+acd72(76)+acd72(74)+acd72(72)+acd72(70)+acd72(68)+2.0&
      &_ki*acd72(62)+acd72(64)+acd72(66)
      acd72(64)=ninjaP+acd72(37)
      acd72(60)=acd72(64)*acd72(60)
      acd72(63)=acd72(38)*acd72(63)
      acd72(65)=acd72(39)*acd72(65)
      acd72(66)=acd72(32)*acd72(9)
      acd72(68)=acd72(33)*acd72(11)
      acd72(70)=acd72(34)*acd72(15)
      acd72(66)=acd72(70)+acd72(68)+acd72(66)
      acd72(66)=acd72(47)*acd72(66)
      acd72(68)=acd72(31)*acd72(5)
      acd72(70)=acd72(35)*acd72(7)
      acd72(72)=acd72(36)*acd72(13)
      acd72(68)=acd72(72)+acd72(70)-acd72(68)
      acd72(68)=acd72(49)*acd72(68)
      acd72(70)=2.0_ki*acd72(17)
      acd72(72)=acd72(6)*acd72(70)
      acd72(74)=acd72(19)*acd72(1)
      acd72(67)=-acd72(67)+acd72(72)+acd72(74)
      acd72(67)=acd72(40)*acd72(67)
      acd72(72)=acd72(8)*acd72(70)
      acd72(74)=acd72(24)*acd72(1)
      acd72(72)=acd72(75)+acd72(72)+acd72(74)
      acd72(72)=acd72(41)*acd72(72)
      acd72(74)=-acd72(10)*acd72(70)
      acd72(75)=acd72(26)*acd72(3)
      acd72(69)=acd72(69)+acd72(74)+acd72(75)
      acd72(69)=acd72(42)*acd72(69)
      acd72(74)=acd72(12)*acd72(70)
      acd72(75)=acd72(27)*acd72(3)
      acd72(71)=acd72(71)+acd72(74)+acd72(75)
      acd72(71)=acd72(43)*acd72(71)
      acd72(74)=acd72(14)*acd72(70)
      acd72(75)=acd72(25)*acd72(1)
      acd72(74)=acd72(77)+acd72(74)+acd72(75)
      acd72(74)=acd72(44)*acd72(74)
      acd72(75)=acd72(16)*acd72(70)
      acd72(76)=-acd72(30)*acd72(3)
      acd72(73)=acd72(73)+acd72(75)+acd72(76)
      acd72(73)=acd72(45)*acd72(73)
      acd72(75)=acd72(1)*acd72(64)
      acd72(76)=acd72(38)*acd72(70)
      acd72(75)=acd72(76)+acd72(75)
      acd72(75)=acd72(2)*acd72(75)
      acd72(64)=acd72(3)*acd72(64)
      acd72(76)=acd72(39)*acd72(70)
      acd72(64)=acd72(76)+acd72(64)
      acd72(64)=acd72(4)*acd72(64)
      acd72(76)=acd72(38)*acd72(3)
      acd72(77)=acd72(39)*acd72(1)
      acd72(76)=acd72(76)+acd72(77)
      acd72(76)=acd72(18)*acd72(76)
      acd72(77)=acd72(38)*acd72(20)
      acd72(78)=acd72(47)*acd72(1)
      acd72(77)=acd72(77)+acd72(78)
      acd72(77)=acd72(21)*acd72(77)
      acd72(78)=acd72(39)*acd72(28)
      acd72(79)=acd72(49)*acd72(3)
      acd72(78)=acd72(78)+acd72(79)
      acd72(78)=acd72(29)*acd72(78)
      acd72(79)=acd72(48)*acd72(23)
      acd72(79)=acd72(50)+acd72(79)
      acd72(79)=acd72(1)*acd72(79)
      acd72(70)=acd72(46)*acd72(70)
      acd72(80)=acd72(51)*acd72(3)
      acd72(81)=acd72(52)*acd72(5)
      acd72(82)=acd72(53)*acd72(20)
      acd72(83)=acd72(54)*acd72(7)
      acd72(84)=acd72(55)*acd72(9)
      acd72(85)=acd72(56)*acd72(11)
      acd72(86)=acd72(57)*acd72(28)
      acd72(87)=acd72(58)*acd72(13)
      acd72(88)=acd72(59)*acd72(15)
      acd72(60)=acd72(88)+acd72(87)+acd72(86)+acd72(85)+acd72(84)+acd72(83)+acd&
      &72(82)+acd72(81)+acd72(80)+acd72(70)+acd72(73)+acd72(74)+acd72(71)+acd72&
      &(69)+acd72(72)+acd72(67)+acd72(78)+acd72(77)+acd72(76)+acd72(75)+acd72(6&
      &4)+acd72(63)+acd72(68)+acd72(66)+acd72(65)+acd72(79)+acd72(60)
      brack(ninjaidxt1mu0)=acd72(62)
      brack(ninjaidxt0mu0)=acd72(60)
      brack(ninjaidxt0mu2)=acd72(61)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d72h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd72h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k4
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
end module     p2_gg_httbar_d72h4l131_qp
