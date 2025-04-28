module     p2_gg_httbar_d147h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d147h4l131.f90
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
      use p2_gg_httbar_abbrevd147h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd147
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd147(1)=dotproduct(ninjaE3,spvak2e2)
      acd147(2)=dotproduct(ninjaE3,spvae2k2)
      acd147(3)=abb147(12)
      acd147(4)=dotproduct(ninjaE3,spvae2l4)
      acd147(5)=abb147(20)
      acd147(6)=dotproduct(ninjaE3,spvae2e1)
      acd147(7)=abb147(53)
      acd147(8)=dotproduct(ninjaE3,spvae2k1)
      acd147(9)=abb147(59)
      acd147(10)=dotproduct(ninjaE3,spvae2l5)
      acd147(11)=abb147(61)
      acd147(12)=dotproduct(ninjaE3,spvak1e2)
      acd147(13)=abb147(13)
      acd147(14)=dotproduct(ninjaE3,spvae1e2)
      acd147(15)=abb147(14)
      acd147(16)=dotproduct(ninjaE3,spval5e2)
      acd147(17)=abb147(25)
      acd147(18)=dotproduct(ninjaE3,spval4e2)
      acd147(19)=abb147(28)
      acd147(20)=acd147(5)*acd147(1)
      acd147(21)=acd147(13)*acd147(12)
      acd147(22)=acd147(15)*acd147(14)
      acd147(23)=acd147(17)*acd147(16)
      acd147(24)=acd147(19)*acd147(18)
      acd147(20)=acd147(24)+acd147(23)+acd147(22)+acd147(21)+acd147(20)
      acd147(20)=acd147(4)*acd147(20)
      acd147(21)=acd147(3)*acd147(2)
      acd147(22)=acd147(7)*acd147(6)
      acd147(23)=-acd147(9)*acd147(8)
      acd147(24)=acd147(11)*acd147(10)
      acd147(21)=acd147(24)+acd147(23)+acd147(22)+acd147(21)
      acd147(21)=acd147(1)*acd147(21)
      acd147(20)=acd147(20)+acd147(21)
      brack(ninjaidxt2mu0)=acd147(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd147h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(65) :: acd147
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd147(1)=dotproduct(ninjaE3,spvak2e2)
      acd147(2)=dotproduct(ninjaE4,spvae2k2)
      acd147(3)=abb147(12)
      acd147(4)=dotproduct(ninjaE4,spvae2l4)
      acd147(5)=abb147(20)
      acd147(6)=dotproduct(ninjaE4,spvae2e1)
      acd147(7)=abb147(53)
      acd147(8)=dotproduct(ninjaE4,spvae2k1)
      acd147(9)=abb147(59)
      acd147(10)=dotproduct(ninjaE4,spvae2l5)
      acd147(11)=abb147(61)
      acd147(12)=dotproduct(ninjaE3,spvae2k2)
      acd147(13)=dotproduct(ninjaE4,spvak2e2)
      acd147(14)=dotproduct(ninjaE3,spvak1e2)
      acd147(15)=abb147(13)
      acd147(16)=dotproduct(ninjaE3,spvae2l4)
      acd147(17)=dotproduct(ninjaE4,spvak1e2)
      acd147(18)=dotproduct(ninjaE4,spvae1e2)
      acd147(19)=abb147(14)
      acd147(20)=dotproduct(ninjaE4,spval4e2)
      acd147(21)=abb147(28)
      acd147(22)=dotproduct(ninjaE4,spval5e2)
      acd147(23)=abb147(25)
      acd147(24)=dotproduct(ninjaE3,spvae1e2)
      acd147(25)=dotproduct(ninjaE3,spvae2e1)
      acd147(26)=dotproduct(ninjaE3,spval4e2)
      acd147(27)=dotproduct(ninjaE3,spvae2k1)
      acd147(28)=dotproduct(ninjaE3,spval5e2)
      acd147(29)=dotproduct(ninjaE3,spvae2l5)
      acd147(30)=dotproduct(ninjaA,spvak2e2)
      acd147(31)=dotproduct(ninjaA,spvae2k2)
      acd147(32)=dotproduct(ninjaA,spvak1e2)
      acd147(33)=dotproduct(ninjaA,spvae2l4)
      acd147(34)=dotproduct(ninjaA,spvae1e2)
      acd147(35)=dotproduct(ninjaA,spvae2e1)
      acd147(36)=dotproduct(ninjaA,spval4e2)
      acd147(37)=dotproduct(ninjaA,spvae2k1)
      acd147(38)=dotproduct(ninjaA,spval5e2)
      acd147(39)=dotproduct(ninjaA,spvae2l5)
      acd147(40)=abb147(17)
      acd147(41)=abb147(36)
      acd147(42)=abb147(21)
      acd147(43)=abb147(31)
      acd147(44)=abb147(22)
      acd147(45)=abb147(16)
      acd147(46)=abb147(18)
      acd147(47)=abb147(19)
      acd147(48)=abb147(43)
      acd147(49)=abb147(29)
      acd147(50)=abb147(15)
      acd147(51)=acd147(23)*acd147(22)
      acd147(52)=acd147(21)*acd147(20)
      acd147(53)=acd147(19)*acd147(18)
      acd147(54)=acd147(15)*acd147(17)
      acd147(55)=acd147(13)*acd147(5)
      acd147(51)=acd147(51)+acd147(53)+acd147(54)+acd147(52)+acd147(55)
      acd147(51)=acd147(51)*acd147(16)
      acd147(52)=acd147(11)*acd147(10)
      acd147(53)=acd147(9)*acd147(8)
      acd147(54)=acd147(7)*acd147(6)
      acd147(55)=acd147(3)*acd147(2)
      acd147(56)=acd147(4)*acd147(5)
      acd147(52)=acd147(56)+acd147(52)-acd147(53)+acd147(54)+acd147(55)
      acd147(52)=acd147(52)*acd147(1)
      acd147(53)=acd147(11)*acd147(29)
      acd147(54)=acd147(9)*acd147(27)
      acd147(55)=acd147(7)*acd147(25)
      acd147(56)=acd147(3)*acd147(12)
      acd147(53)=acd147(56)+acd147(55)+acd147(53)-acd147(54)
      acd147(54)=acd147(53)*acd147(13)
      acd147(55)=acd147(23)*acd147(28)
      acd147(56)=acd147(21)*acd147(26)
      acd147(57)=acd147(19)*acd147(24)
      acd147(58)=acd147(15)*acd147(14)
      acd147(55)=acd147(55)+acd147(56)+acd147(57)+acd147(58)
      acd147(56)=acd147(55)*acd147(4)
      acd147(51)=acd147(54)+acd147(56)+acd147(51)+acd147(52)
      acd147(52)=acd147(11)*acd147(39)
      acd147(54)=acd147(9)*acd147(37)
      acd147(56)=acd147(7)*acd147(35)
      acd147(57)=acd147(3)*acd147(31)
      acd147(58)=acd147(33)*acd147(5)
      acd147(52)=acd147(52)-acd147(54)+acd147(56)+acd147(57)+acd147(58)+acd147(&
      &40)
      acd147(54)=acd147(1)*acd147(52)
      acd147(55)=acd147(33)*acd147(55)
      acd147(53)=acd147(30)*acd147(53)
      acd147(56)=acd147(23)*acd147(38)
      acd147(57)=acd147(21)*acd147(36)
      acd147(58)=acd147(19)*acd147(34)
      acd147(59)=acd147(15)*acd147(32)
      acd147(56)=acd147(56)+acd147(57)+acd147(58)+acd147(59)+acd147(43)
      acd147(57)=acd147(30)*acd147(5)
      acd147(57)=acd147(57)+acd147(56)
      acd147(57)=acd147(16)*acd147(57)
      acd147(58)=acd147(29)*acd147(49)
      acd147(59)=acd147(28)*acd147(48)
      acd147(60)=acd147(27)*acd147(47)
      acd147(61)=acd147(26)*acd147(46)
      acd147(62)=acd147(25)*acd147(45)
      acd147(63)=acd147(24)*acd147(44)
      acd147(64)=acd147(14)*acd147(42)
      acd147(65)=acd147(12)*acd147(41)
      acd147(53)=acd147(54)+acd147(57)+acd147(53)+acd147(55)+acd147(65)+acd147(&
      &64)+acd147(63)+acd147(62)+acd147(61)+acd147(60)+acd147(58)+acd147(59)
      acd147(54)=ninjaP*acd147(51)
      acd147(52)=acd147(30)*acd147(52)
      acd147(55)=acd147(33)*acd147(56)
      acd147(56)=acd147(39)*acd147(49)
      acd147(57)=acd147(38)*acd147(48)
      acd147(58)=acd147(37)*acd147(47)
      acd147(59)=acd147(36)*acd147(46)
      acd147(60)=acd147(35)*acd147(45)
      acd147(61)=acd147(34)*acd147(44)
      acd147(62)=acd147(32)*acd147(42)
      acd147(63)=acd147(31)*acd147(41)
      acd147(52)=acd147(54)+acd147(52)+acd147(55)+acd147(63)+acd147(62)+acd147(&
      &61)+acd147(60)+acd147(59)+acd147(58)+acd147(57)+acd147(50)+acd147(56)
      brack(ninjaidxt1mu0)=acd147(53)
      brack(ninjaidxt0mu0)=acd147(52)
      brack(ninjaidxt0mu2)=acd147(51)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d147h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd147h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k4
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
end module     p2_gg_httbar_d147h4l131
