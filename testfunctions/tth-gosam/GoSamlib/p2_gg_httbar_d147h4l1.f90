module     p2_gg_httbar_d147h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d147h4l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd147h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc147(30)
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval4e2
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      acc147(1)=abb147(12)
      acc147(2)=abb147(13)
      acc147(3)=abb147(14)
      acc147(4)=abb147(15)
      acc147(5)=abb147(16)
      acc147(6)=abb147(17)
      acc147(7)=abb147(18)
      acc147(8)=abb147(19)
      acc147(9)=abb147(20)
      acc147(10)=abb147(21)
      acc147(11)=abb147(22)
      acc147(12)=abb147(25)
      acc147(13)=abb147(28)
      acc147(14)=abb147(29)
      acc147(15)=abb147(31)
      acc147(16)=abb147(36)
      acc147(17)=abb147(43)
      acc147(18)=abb147(53)
      acc147(19)=abb147(59)
      acc147(20)=abb147(61)
      acc147(21)=acc147(1)*Qspvae2k2
      acc147(22)=acc147(9)*Qspvae2l4
      acc147(23)=acc147(18)*Qspvae2e1
      acc147(24)=-acc147(19)*Qspvae2k1
      acc147(25)=acc147(20)*Qspvae2l5
      acc147(21)=acc147(25)+acc147(24)+acc147(23)+acc147(22)+acc147(6)+acc147(2&
      &1)
      acc147(21)=Qspvak2e2*acc147(21)
      acc147(22)=acc147(2)*Qspvak1e2
      acc147(23)=acc147(3)*Qspvae1e2
      acc147(24)=acc147(12)*Qspval5e2
      acc147(25)=acc147(13)*Qspval4e2
      acc147(22)=acc147(15)+acc147(25)+acc147(24)+acc147(23)+acc147(22)
      acc147(22)=Qspvae2l4*acc147(22)
      acc147(23)=acc147(5)*Qspvae2e1
      acc147(24)=acc147(7)*Qspval4e2
      acc147(25)=acc147(8)*Qspvae2k1
      acc147(26)=acc147(10)*Qspvak1e2
      acc147(27)=acc147(11)*Qspvae1e2
      acc147(28)=acc147(14)*Qspvae2l5
      acc147(29)=acc147(16)*Qspvae2k2
      acc147(30)=acc147(17)*Qspval5e2
      brack=acc147(4)+acc147(21)+acc147(22)+acc147(23)+acc147(24)+acc147(25)+ac&
      &c147(26)+acc147(27)+acc147(28)+acc147(29)+acc147(30)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d147h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd147h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d147
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d147 = 0.0_ki
      d147 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d147, ki), aimag(d147), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d147h4l1
