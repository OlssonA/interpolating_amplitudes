module     p2_gg_httbar_d10h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d10h12l1.f90
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
      use p2_gg_httbar_abbrevd10h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc10(28)
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      acc10(1)=abb10(9)
      acc10(2)=abb10(10)
      acc10(3)=abb10(11)
      acc10(4)=abb10(12)
      acc10(5)=abb10(13)
      acc10(6)=abb10(14)
      acc10(7)=abb10(15)
      acc10(8)=abb10(16)
      acc10(9)=abb10(17)
      acc10(10)=abb10(18)
      acc10(11)=abb10(19)
      acc10(12)=abb10(20)
      acc10(13)=abb10(21)
      acc10(14)=abb10(22)
      acc10(15)=abb10(25)
      acc10(16)=abb10(26)
      acc10(17)=abb10(27)
      acc10(18)=abb10(28)
      acc10(19)=acc10(4)*Qspvak1l3
      acc10(20)=acc10(17)*Qspvak1l5
      acc10(21)=acc10(18)*Qspvak1l4
      acc10(19)=acc10(21)+acc10(20)+acc10(10)+acc10(19)
      acc10(19)=Qspvak2k1*acc10(19)
      acc10(20)=acc10(3)*Qspvak2l3
      acc10(21)=acc10(9)*Qspvak2l5
      acc10(22)=acc10(14)*Qspvak2l4
      acc10(20)=acc10(22)+acc10(11)+acc10(21)+acc10(20)
      acc10(20)=Qspk2*acc10(20)
      acc10(21)=acc10(13)*Qspvak2l5
      acc10(21)=acc10(12)+acc10(21)
      acc10(21)=Qspval3k2*acc10(21)
      acc10(22)=-acc10(13)*Qspvak1l5
      acc10(22)=acc10(16)+acc10(22)
      acc10(22)=Qspval3k1*acc10(22)
      acc10(23)=acc10(1)*Qspvak2l3
      acc10(24)=acc10(2)*Qspvak1l4
      acc10(25)=acc10(5)*Qspvak2l5
      acc10(26)=acc10(6)*Qspvak1l3
      acc10(27)=acc10(8)*Qspvak2l4
      acc10(28)=acc10(15)*Qspvak1l5
      brack=acc10(7)+acc10(19)+acc10(20)+acc10(21)+acc10(22)+acc10(23)+acc10(24&
      &)+acc10(25)+acc10(26)+acc10(27)+acc10(28)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d10h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd10h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d10
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d10 = 0.0_ki
      d10 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d10, ki), aimag(d10), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d10h12l1
