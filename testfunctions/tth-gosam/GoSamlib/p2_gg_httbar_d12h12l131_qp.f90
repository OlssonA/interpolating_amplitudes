module     p2_gg_httbar_d12h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d12h12l131_qp.f90
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
      use p2_gg_httbar_abbrevd12h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(21) :: acd12
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd12(1)=dotproduct(k2,ninjaE3)
      acd12(2)=dotproduct(ninjaE3,spvak2l3)
      acd12(3)=abb12(11)
      acd12(4)=dotproduct(ninjaE3,spvak2l4)
      acd12(5)=abb12(17)
      acd12(6)=dotproduct(ninjaE3,spvak2l5)
      acd12(7)=abb12(22)
      acd12(8)=dotproduct(ninjaE3,spvak1l3)
      acd12(9)=dotproduct(ninjaE3,spvak2k1)
      acd12(10)=abb12(12)
      acd12(11)=dotproduct(ninjaE3,spvak1l4)
      acd12(12)=abb12(28)
      acd12(13)=dotproduct(ninjaE3,spvak1l5)
      acd12(14)=abb12(27)
      acd12(15)=dotproduct(ninjaE3,spval3k2)
      acd12(16)=abb12(21)
      acd12(17)=dotproduct(ninjaE3,spval3k1)
      acd12(18)=acd12(3)*acd12(2)
      acd12(19)=acd12(5)*acd12(4)
      acd12(20)=acd12(7)*acd12(6)
      acd12(18)=acd12(20)+acd12(18)+acd12(19)
      acd12(18)=acd12(1)*acd12(18)
      acd12(19)=acd12(10)*acd12(8)
      acd12(20)=acd12(12)*acd12(11)
      acd12(21)=acd12(14)*acd12(13)
      acd12(19)=acd12(21)+acd12(20)+acd12(19)
      acd12(19)=acd12(9)*acd12(19)
      acd12(20)=acd12(15)*acd12(4)
      acd12(21)=-acd12(17)*acd12(11)
      acd12(20)=acd12(21)+acd12(20)
      acd12(20)=acd12(16)*acd12(20)
      acd12(18)=acd12(19)+acd12(18)+acd12(20)
      brack(ninjaidxt2mu0)=acd12(18)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd12h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(64) :: acd12
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd12(1)=dotproduct(k2,ninjaE3)
      acd12(2)=dotproduct(ninjaE4,spvak2l3)
      acd12(3)=abb12(11)
      acd12(4)=dotproduct(ninjaE4,spvak2l5)
      acd12(5)=abb12(22)
      acd12(6)=dotproduct(ninjaE4,spvak2l4)
      acd12(7)=abb12(17)
      acd12(8)=dotproduct(k2,ninjaE4)
      acd12(9)=dotproduct(ninjaE3,spvak2l3)
      acd12(10)=dotproduct(ninjaE3,spvak2l5)
      acd12(11)=dotproduct(ninjaE3,spvak2l4)
      acd12(12)=dotproduct(ninjaE3,spvak1l4)
      acd12(13)=dotproduct(ninjaE4,spvak2k1)
      acd12(14)=abb12(28)
      acd12(15)=dotproduct(ninjaE4,spval3k1)
      acd12(16)=abb12(21)
      acd12(17)=dotproduct(ninjaE3,spvak1l3)
      acd12(18)=abb12(12)
      acd12(19)=dotproduct(ninjaE3,spvak2k1)
      acd12(20)=dotproduct(ninjaE4,spvak1l4)
      acd12(21)=dotproduct(ninjaE4,spvak1l3)
      acd12(22)=dotproduct(ninjaE4,spvak1l5)
      acd12(23)=abb12(27)
      acd12(24)=dotproduct(ninjaE4,spval3k2)
      acd12(25)=dotproduct(ninjaE3,spval3k2)
      acd12(26)=dotproduct(ninjaE3,spval3k1)
      acd12(27)=dotproduct(ninjaE3,spvak1l5)
      acd12(28)=dotproduct(k2,ninjaA)
      acd12(29)=dotproduct(ninjaA,spvak2l3)
      acd12(30)=dotproduct(ninjaA,spvak2l5)
      acd12(31)=dotproduct(ninjaA,spvak2l4)
      acd12(32)=abb12(19)
      acd12(33)=dotproduct(ninjaA,spvak1l4)
      acd12(34)=dotproduct(ninjaA,spvak1l3)
      acd12(35)=dotproduct(ninjaA,spvak2k1)
      acd12(36)=dotproduct(ninjaA,spval3k2)
      acd12(37)=dotproduct(ninjaA,spval3k1)
      acd12(38)=dotproduct(ninjaA,spvak1l5)
      acd12(39)=abb12(9)
      acd12(40)=abb12(10)
      acd12(41)=abb12(14)
      acd12(42)=abb12(18)
      acd12(43)=abb12(13)
      acd12(44)=abb12(16)
      acd12(45)=abb12(20)
      acd12(46)=abb12(26)
      acd12(47)=abb12(25)
      acd12(48)=abb12(15)
      acd12(49)=acd12(20)*acd12(26)
      acd12(50)=acd12(6)*acd12(25)
      acd12(51)=acd12(12)*acd12(15)
      acd12(52)=acd12(11)*acd12(24)
      acd12(49)=-acd12(50)+acd12(49)+acd12(51)-acd12(52)
      acd12(49)=acd12(49)*acd12(16)
      acd12(50)=acd12(7)*acd12(6)
      acd12(51)=acd12(5)*acd12(4)
      acd12(52)=acd12(3)*acd12(2)
      acd12(50)=acd12(52)+acd12(50)+acd12(51)
      acd12(50)=acd12(50)*acd12(1)
      acd12(51)=acd12(23)*acd12(22)
      acd12(52)=acd12(18)*acd12(21)
      acd12(53)=acd12(14)*acd12(20)
      acd12(51)=acd12(53)+acd12(51)+acd12(52)
      acd12(51)=acd12(51)*acd12(19)
      acd12(52)=acd12(7)*acd12(11)
      acd12(53)=acd12(5)*acd12(10)
      acd12(54)=acd12(3)*acd12(9)
      acd12(52)=acd12(54)+acd12(52)+acd12(53)
      acd12(53)=acd12(52)*acd12(8)
      acd12(54)=acd12(23)*acd12(27)
      acd12(55)=acd12(18)*acd12(17)
      acd12(54)=acd12(54)+acd12(55)
      acd12(55)=acd12(12)*acd12(14)
      acd12(55)=acd12(55)+acd12(54)
      acd12(55)=acd12(55)*acd12(13)
      acd12(49)=acd12(50)+acd12(55)+acd12(53)+acd12(51)-acd12(49)
      acd12(50)=-acd12(33)*acd12(26)
      acd12(51)=acd12(31)*acd12(25)
      acd12(53)=-acd12(12)*acd12(37)
      acd12(55)=acd12(11)*acd12(36)
      acd12(50)=acd12(55)+acd12(53)+acd12(50)+acd12(51)
      acd12(50)=acd12(16)*acd12(50)
      acd12(51)=acd12(28)*acd12(52)
      acd12(52)=acd12(23)*acd12(38)
      acd12(53)=acd12(18)*acd12(34)
      acd12(55)=acd12(14)*acd12(33)
      acd12(52)=acd12(52)+acd12(53)+acd12(55)+acd12(42)
      acd12(53)=acd12(19)*acd12(52)
      acd12(55)=acd12(7)*acd12(31)
      acd12(55)=acd12(55)+acd12(32)
      acd12(56)=acd12(3)*acd12(29)
      acd12(57)=acd12(5)*acd12(30)
      acd12(56)=acd12(56)+acd12(57)+acd12(55)
      acd12(56)=acd12(1)*acd12(56)
      acd12(54)=acd12(35)*acd12(54)
      acd12(57)=acd12(27)*acd12(47)
      acd12(58)=acd12(26)*acd12(46)
      acd12(59)=acd12(25)*acd12(45)
      acd12(60)=acd12(17)*acd12(41)
      acd12(61)=acd12(10)*acd12(43)
      acd12(62)=acd12(9)*acd12(39)
      acd12(63)=acd12(14)*acd12(35)
      acd12(63)=acd12(40)+acd12(63)
      acd12(63)=acd12(12)*acd12(63)
      acd12(64)=acd12(11)*acd12(44)
      acd12(50)=acd12(50)+acd12(56)+acd12(53)+acd12(64)+acd12(63)+acd12(62)+acd&
      &12(61)+acd12(60)+acd12(59)+acd12(57)+acd12(58)+acd12(51)+acd12(54)
      acd12(51)=ninjaP*acd12(49)
      acd12(52)=acd12(35)*acd12(52)
      acd12(53)=-acd12(33)*acd12(37)
      acd12(54)=acd12(31)*acd12(36)
      acd12(53)=acd12(53)+acd12(54)
      acd12(53)=acd12(16)*acd12(53)
      acd12(54)=acd12(5)*acd12(28)
      acd12(54)=acd12(54)+acd12(43)
      acd12(54)=acd12(30)*acd12(54)
      acd12(56)=acd12(3)*acd12(28)
      acd12(56)=acd12(56)+acd12(39)
      acd12(56)=acd12(29)*acd12(56)
      acd12(55)=acd12(28)*acd12(55)
      acd12(57)=acd12(38)*acd12(47)
      acd12(58)=acd12(37)*acd12(46)
      acd12(59)=acd12(36)*acd12(45)
      acd12(60)=acd12(34)*acd12(41)
      acd12(61)=acd12(33)*acd12(40)
      acd12(62)=acd12(31)*acd12(44)
      acd12(51)=acd12(51)+acd12(53)+acd12(62)+acd12(61)+acd12(60)+acd12(59)+acd&
      &12(58)+acd12(48)+acd12(57)+acd12(52)+acd12(55)+acd12(56)+acd12(54)
      brack(ninjaidxt1mu0)=acd12(50)
      brack(ninjaidxt0mu0)=acd12(51)
      brack(ninjaidxt0mu2)=acd12(49)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d12h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd12h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
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
end module     p2_gg_httbar_d12h12l131_qp
