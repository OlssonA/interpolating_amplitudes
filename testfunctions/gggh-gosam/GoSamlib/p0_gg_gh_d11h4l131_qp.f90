module     p0_gg_gh_d11h4l131_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d11h4l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(5) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(ninjaE3,spvak1k2)
      acd11(2)=dotproduct(ninjaE3,spvak2k3)
      acd11(3)=dotproduct(ninjaE3,spvak3k2)
      acd11(4)=abb11(7)
      acd11(5)=acd11(4)*acd11(3)*acd11(2)*acd11(1)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd11(5)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd11h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(56) :: acd11
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd11(1)=dotproduct(ninjaE3,spvak1k2)
      acd11(2)=dotproduct(ninjaE3,spvak2k3)
      acd11(3)=dotproduct(ninjaE4,spvak3k2)
      acd11(4)=abb11(7)
      acd11(5)=dotproduct(ninjaE3,spvak3k2)
      acd11(6)=dotproduct(ninjaE4,spvak2k3)
      acd11(7)=abb11(29)
      acd11(8)=dotproduct(ninjaE4,spvak1k2)
      acd11(9)=dotproduct(k1,ninjaE3)
      acd11(10)=abb11(9)
      acd11(11)=dotproduct(k2,ninjaE3)
      acd11(12)=abb11(21)
      acd11(13)=abb11(12)
      acd11(14)=abb11(20)
      acd11(15)=dotproduct(ninjaE3,spval4k2)
      acd11(16)=abb11(27)
      acd11(17)=dotproduct(ninjaE3,spvak1k3)
      acd11(18)=abb11(26)
      acd11(19)=dotproduct(k3,ninjaE3)
      acd11(20)=abb11(19)
      acd11(21)=dotproduct(ninjaA,ninjaE3)
      acd11(22)=dotproduct(ninjaA,spvak1k2)
      acd11(23)=dotproduct(ninjaA,spvak2k3)
      acd11(24)=dotproduct(ninjaA,spvak3k2)
      acd11(25)=abb11(15)
      acd11(26)=abb11(11)
      acd11(27)=abb11(13)
      acd11(28)=dotproduct(ninjaE3,spvak1l4)
      acd11(29)=abb11(23)
      acd11(30)=dotproduct(k1,ninjaA)
      acd11(31)=dotproduct(k2,ninjaA)
      acd11(32)=dotproduct(ninjaA,spval4k2)
      acd11(33)=dotproduct(ninjaA,spvak1k3)
      acd11(34)=abb11(16)
      acd11(35)=dotproduct(k3,ninjaA)
      acd11(36)=dotproduct(ninjaA,ninjaA)
      acd11(37)=abb11(24)
      acd11(38)=dotproduct(ninjaA,spvak1l4)
      acd11(39)=abb11(8)
      acd11(40)=abb11(14)
      acd11(41)=abb11(25)
      acd11(42)=abb11(17)
      acd11(43)=acd11(5)*acd11(4)
      acd11(44)=acd11(43)*acd11(6)
      acd11(45)=acd11(4)*acd11(2)
      acd11(46)=acd11(3)*acd11(45)
      acd11(46)=acd11(44)+acd11(7)+acd11(46)
      acd11(46)=acd11(1)*acd11(46)
      acd11(47)=acd11(5)*acd11(8)*acd11(45)
      acd11(46)=acd11(47)+acd11(46)
      acd11(47)=acd11(20)*acd11(19)
      acd11(48)=acd11(15)*acd11(26)
      acd11(49)=acd11(17)*acd11(27)
      acd11(50)=2.0_ki*acd11(21)
      acd11(51)=acd11(50)*acd11(7)
      acd11(47)=acd11(47)+acd11(48)+acd11(49)+acd11(51)
      acd11(48)=acd11(11)*acd11(13)
      acd11(49)=acd11(24)*acd11(45)
      acd11(43)=acd11(23)*acd11(43)
      acd11(51)=acd11(1)*acd11(25)
      acd11(43)=acd11(51)+acd11(43)+acd11(49)+acd11(48)+acd11(47)
      acd11(43)=acd11(1)*acd11(43)
      acd11(48)=acd11(29)*acd11(28)
      acd11(49)=acd11(10)*acd11(9)
      acd11(45)=acd11(45)*acd11(22)
      acd11(51)=acd11(7)*acd11(17)
      acd11(45)=acd11(48)+acd11(49)+acd11(45)-acd11(51)
      acd11(48)=acd11(11)*acd11(14)
      acd11(48)=acd11(48)+acd11(45)
      acd11(48)=acd11(5)*acd11(48)
      acd11(49)=acd11(15)*acd11(16)
      acd11(51)=acd11(17)*acd11(18)
      acd11(52)=acd11(11)*acd11(12)
      acd11(49)=acd11(52)+acd11(49)+acd11(51)
      acd11(49)=acd11(11)*acd11(49)
      acd11(43)=acd11(43)+acd11(49)+acd11(48)
      acd11(48)=acd11(24)*acd11(23)
      acd11(49)=acd11(2)*ninjaP
      acd11(51)=acd11(3)*acd11(49)
      acd11(48)=acd11(48)+acd11(51)
      acd11(48)=acd11(4)*acd11(48)
      acd11(51)=acd11(27)*acd11(33)
      acd11(52)=acd11(26)*acd11(32)
      acd11(53)=acd11(20)*acd11(35)
      acd11(54)=acd11(31)*acd11(13)
      acd11(55)=acd11(36)+ninjaP
      acd11(55)=acd11(7)*acd11(55)
      acd11(56)=acd11(22)*acd11(25)
      acd11(44)=ninjaP*acd11(44)
      acd11(44)=acd11(44)+acd11(48)+2.0_ki*acd11(56)+acd11(55)+acd11(54)+acd11(&
      &53)+acd11(52)+acd11(39)+acd11(51)
      acd11(44)=acd11(1)*acd11(44)
      acd11(48)=acd11(18)*acd11(33)
      acd11(51)=acd11(16)*acd11(32)
      acd11(52)=acd11(31)*acd11(12)
      acd11(53)=acd11(24)*acd11(14)
      acd11(54)=acd11(22)*acd11(13)
      acd11(48)=acd11(54)+acd11(53)+2.0_ki*acd11(52)+acd11(51)+acd11(34)+acd11(&
      &48)
      acd11(48)=acd11(11)*acd11(48)
      acd11(49)=acd11(8)*acd11(49)
      acd11(51)=acd11(22)*acd11(23)
      acd11(49)=acd11(49)+acd11(51)
      acd11(49)=acd11(4)*acd11(49)
      acd11(51)=acd11(29)*acd11(38)
      acd11(52)=acd11(10)*acd11(30)
      acd11(53)=acd11(31)*acd11(14)
      acd11(54)=-acd11(7)*acd11(33)
      acd11(49)=acd11(49)+acd11(54)+acd11(53)+acd11(52)+acd11(40)+acd11(51)
      acd11(49)=acd11(5)*acd11(49)
      acd11(45)=acd11(24)*acd11(45)
      acd11(47)=acd11(22)*acd11(47)
      acd11(50)=acd11(37)*acd11(50)
      acd11(51)=acd11(31)*acd11(16)
      acd11(51)=acd11(41)+acd11(51)
      acd11(51)=acd11(15)*acd11(51)
      acd11(52)=acd11(31)*acd11(18)
      acd11(52)=acd11(42)+acd11(52)
      acd11(52)=acd11(17)*acd11(52)
      acd11(44)=acd11(44)+acd11(49)+acd11(48)+acd11(47)+acd11(52)+acd11(50)+acd&
      &11(51)+acd11(45)
      brack(ninjaidxt1mu0)=acd11(43)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd11(44)
      brack(ninjaidxt0mu2)=acd11(46)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d11h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd11h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_gg_gh_d11h4l131_qp
